from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from comet_ml.exceptions import InterruptedExperiment
import hydra
import logging
import torch
import os
from torchvision.transforms import v2
from omegaconf import DictConfig
from models.vae.model import VAEModel
from tensordict.nn import TensorDictModule
from torch.optim import Adam
from losses.vae_loss import VAELoss
from plotting.vae import plot_vae_samples
from pathlib import Path
from datasets.vae_dataset import VAEDataset
from torch.utils.data import DataLoader
from tensordict import TensorDict
from tqdm import tqdm


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs/", config_name="vae_training")
def main(cfg: DictConfig):
    
    COMET_ML_API_KEY = os.getenv("COMET_ML_API_KEY")
    COMET_ML_PROJECT_NAME = os.getenv("COMET_ML_PROJECT_NAME")
    COMET_ML_WORKSPACE = os.getenv("COMET_ML_WORKSPACE")
    
    experiment = Experiment(
        api_key=COMET_ML_API_KEY,
        project_name=COMET_ML_PROJECT_NAME,
        workspace=COMET_ML_WORKSPACE
    )
    
    experiment.log_parameters(cfg)
    experiment.log_code(folder='src')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    seed = cfg['experiment']['seed']
    
    transform = [
        v2.Resize(size=(cfg['dataset']['height'], cfg['dataset']['width'])),
    ]
    if cfg['dataset']['normalize']:
        transform.append(v2.ToDtype(torch.float32, scale=True))
    transform = v2.Compose(transform)
    
    dataset = VAEDataset(data_path=Path(cfg['dataset']['path']), transform=transform)
    dataset_split_generator = torch.Generator().manual_seed(seed)
    
    val_split = cfg['training']['train_val_split']
    train_split = 1 - val_split
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_split, val_split], dataset_split_generator)

    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['val_batch_size'], shuffle=False)
    
    encoder_params = cfg['model']['encoder']
    decoder_params = cfg['model']['decoder']
    vae_model = VAEModel(input_spatial_dim=cfg['dataset']['height'],
                         input_channels=cfg['model']['input_channels'], 
                         encoder_hidden_dims=encoder_params['hidden_dims'],
                         encoder_hidden_activation=encoder_params['hidden_activation'],
                         encoder_hidden_kernels=encoder_params['hidden_kernels'],
                         encoder_hidden_strides=encoder_params['hidden_strides'],
                         encoder_hidden_paddings=encoder_params['hidden_paddings'],
                         encoder_use_batch_norm=encoder_params['use_batch_norm'],
                         encoder_leaky_relu_neg_slope=encoder_params['leaky_relu_neg_slope'],
                         latent_dim=cfg['model']['latent_dim'],
                         decoder_hidden_dims=decoder_params['hidden_dims'],
                         decoder_hidden_activation=decoder_params['hidden_activation'],
                         decoder_hidden_kernels=decoder_params['hidden_kernels'],
                         decoder_hidden_strides=decoder_params['hidden_strides'],
                         decoder_hidden_paddings=decoder_params['hidden_paddings'],
                         decoder_output_kernel=decoder_params['output_kernel'],
                         decoder_output_stride=decoder_params['output_stride'],
                         decoder_output_padding=decoder_params['output_padding'],
                         decoder_use_batch_norm=decoder_params['use_batch_norm']).to(device)
    vae_model = TensorDictModule(vae_model, in_keys=["pixels_transformed"], out_keys=["q_z", "p_x"])
    
    training_steps = cfg['training']['epochs'] * cfg['training']['train_batch_size']
    vae_loss = VAELoss(vae_model,
                       beta=cfg['training']['kl_divergence_beta'],
                       training_steps=training_steps,
                       annealing_strategy=cfg['training']['kl_divergence_annealing_strategy'],
                       annealing_cycles=cfg['training']['kl_divergence_annealing_cycles'],
                       annealing_ratio=cfg['training']['kl_divergence_annealing_ratio'],
                       reconstruction_loss=cfg['training']['reconstruction_loss'])
    
    if cfg['model']['optimizer']['name'] == "adam":
        optimizer = Adam(vae_loss.parameters(), lr=cfg['model']['optimizer']['lr'])
    else:
        raise ValueError(f"Unknown optimizer name: '{cfg['model']['optimizer']['name']}'")
    
    try:
        best_val_loss = -1
        best_model_path = None
        train_step = 0
        train_prefix = "train_"
        val_prefix = "val_"
        hydra_output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        val_model_save_path = hydra_output_path / Path(cfg['model']['save_dir'])
        val_model_save_path.mkdir(parents=True, exist_ok=True)
        logger.info("Starting training...")
        for epoch in tqdm(range(cfg['training']['epochs'])):
            logger.info(f"Training Epoch {epoch}")
            train_losses = []
            train_mean_reconstruction_losses = []
            train_mean_kl_divergence_losses = []
            train_mean_kl_divergences = []
            for X_train in train_loader:
                vae_model.train()
                X_train = X_train.to(device)
                X_train = TensorDict(
                    source={
                        "pixels_transformed": X_train
                    },
                    batch_size=[X_train.shape[0]]
                )
                
                loss_result = vae_loss(X_train)
                
                train_losses.append(loss_result['loss'].item())
                train_mean_reconstruction_losses.append(loss_result['mean_reconstruction_loss'])
                train_mean_kl_divergence_losses.append(loss_result['mean_kl_divergence_loss'])
                train_mean_kl_divergences.append(loss_result['mean_kl_divergence'])
                experiment.log_metric(f"{train_prefix}loss", loss_result['loss'].item(), step=train_step)
                experiment.log_metric(f"{train_prefix}mean_reconstruction_loss", loss_result['mean_reconstruction_loss'], step=train_step)
                experiment.log_metric(f"{train_prefix}mean_kl_divergence_loss", loss_result['mean_kl_divergence_loss'], step=train_step)
                experiment.log_metric(f"{train_prefix}mean_kl_divergence", loss_result['mean_kl_divergence'], step=train_step)
                experiment.log_metric(f"{train_prefix}kl_loss_weight", loss_result['kl_loss_weight'], step=train_step)
                
                if train_step % cfg['logging']['val_image_log_steps_interval'] == 0:
                    logger.info("Logging reconstructed samples...")
                    vae_model.eval()
                    random_eval_indxes = torch.randint(0, len(val_dataset), (cfg['logging']['val_num_preview_samples'],))
                    random_eval_samples = []
                    for random_eval_indx in random_eval_indxes:
                        random_eval_samples.append(dataset[random_eval_indx])
                    random_eval_samples = torch.stack(random_eval_samples).to(device)
                    random_eval_samples = TensorDict(
                        source={
                            "pixels_transformed": random_eval_samples
                        },
                        batch_size=[random_eval_samples.shape[0]]
                    )
                    reconstructed_samples_fig = plot_vae_samples(model=vae_model, samples=random_eval_samples, loc=0., scale=1.)
                    experiment.log_image(reconstructed_samples_fig, name=f"{val_prefix}reconstructed_samples_step_{train_step}", step=train_step)
                
                optimizer.zero_grad()
                loss_result['loss'].backward()
                optimizer.step()
                
                train_step += 1
            
            experiment.log_metric(f"{train_prefix}loss", torch.tensor(train_losses).mean().item(), epoch=epoch)
            experiment.log_metric(f"{train_prefix}mean_reconstruction_loss", torch.tensor(train_mean_reconstruction_losses).mean().item(), epoch=epoch)
            experiment.log_metric(f"{train_prefix}mean_kl_divergence_loss", torch.tensor(train_mean_kl_divergence_losses).mean().item(), epoch=epoch)
            experiment.log_metric(f"{train_prefix}mean_kl_divergence", torch.tensor(train_mean_kl_divergences).mean().item(), epoch=epoch)
            
            val_losses = []
            val_mean_reconstruction_losses = []
            val_mean_kl_divergence_losses = []
            val_mean_kl_divergences = []
            vae_model.eval()
            logger.info("Evaluating model...")
            with torch.no_grad():
                for X_val in val_loader:
                    X_val = X_val.to(device)
                    X_val = TensorDict(
                        source={
                            "pixels_transformed": X_val   
                        },
                        batch_size=[X_val.shape[0]]
                    )
                    
                    loss_result = vae_loss(X_val)
                    
                    val_losses.append(loss_result['loss'].item())
                    val_mean_reconstruction_losses.append(loss_result['mean_reconstruction_loss'])
                    val_mean_kl_divergence_losses.append(loss_result['mean_kl_divergence_loss'])
                    val_mean_kl_divergences.append(loss_result['mean_kl_divergence'])
            
            val_epoch_mean_loss = torch.tensor(val_losses).mean().item()
            if val_epoch_mean_loss > best_val_loss:
                logger.info("Saving best model...")
                best_val_loss = val_epoch_mean_loss
                [f.unlink() for f in val_model_save_path.glob("*") if f.is_file()] 
                best_model_path = val_model_save_path / f"model_val_loss_{best_val_loss:.4f}.pt"
                torch.save(vae_model.state_dict(), best_model_path)
            
            experiment.log_metric(f"{val_prefix}loss", val_epoch_mean_loss, epoch=epoch)
            experiment.log_metric(f"{val_prefix}mean_reconstruction_loss", torch.tensor(val_mean_reconstruction_losses).mean().item(), epoch=epoch)
            experiment.log_metric(f"{val_prefix}mean_kl_divergence_loss", torch.tensor(val_mean_kl_divergence_losses).mean().item(), epoch=epoch)
            experiment.log_metric(f"{val_prefix}mean_kl_divergence", torch.tensor(val_mean_kl_divergences).mean().item(), epoch=epoch)
    except InterruptedExperiment as exc:
        experiment.log_other("status", str(exc))
        logger.info("Experiment interrupted!")
    
    logger.info("Training done!")
    logger.info("Saving model...")
    log_model(experiment, vae_model, model_name="vae_model")


if __name__ == "__main__":
    main()
