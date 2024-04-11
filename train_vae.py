import math
from comet_ml import Experiment
from comet_ml import Artifact
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@hydra.main(version_base=None, config_path="configs/", config_name="vae_training")
def main(cfg: DictConfig):
    is_cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA AVAILABLE: {is_cuda_available}")
    
    COMET_ML_API_KEY = os.getenv("COMET_ML_API_KEY")
    COMET_ML_PROJECT_NAME = os.getenv("COMET_ML_PROJECT_NAME")
    COMET_ML_WORKSPACE = os.getenv("COMET_ML_WORKSPACE")
    
    experiment = create_experiment(api_key=COMET_ML_API_KEY, project_name=COMET_ML_PROJECT_NAME, workspasce=COMET_ML_WORKSPACE)
    
    experiment.log_parameters(cfg)
    experiment.log_code(folder='src')
    experiment.log_other("cuda_available", is_cuda_available)
    
    dataset = create_dataset(cfg)
    experiment.log_dataset_info(name=Path(cfg['dataset']['path']).stem, path=str(Path(cfg['dataset']['path'])))
    experiment.log_other("dataset_size", len(dataset))
    
    train_dataset, val_dataset = split_train_val_dataset(dataset, cfg)
    experiment.log_other("train_dataset_size", len(train_dataset))
    experiment.log_other("val_dataset_size", len(val_dataset))

    train_loader = create_train_data_loader(train_dataset, cfg)
    val_loader = create_val_data_loader(val_dataset, cfg)
    
    device = torch.device("cuda" if is_cuda_available else "cpu")
    vae_model = create_vae_model(cfg, device)
    
    training_steps = int(cfg['training']['epochs'] * math.ceil(len(train_dataset) / cfg['training']['train_batch_size']))
    
    vae_loss = create_vae_loss(vae_model, cfg, training_steps)
    
    optimizer = create_optimizer(vae_loss, cfg)
    
    best_val_loss, best_model_path = train_model(vae_model, vae_loss, optimizer, train_loader, val_loader, cfg, experiment, device, val_dataset, logger, training_steps)
    
    experiment.log_metric("val_loss_best", best_val_loss)
    
    save_model(experiment, best_model_path, logger, cfg, vae_model)

    log_log_file(experiment)

    cleanup_resources(dataset, experiment, logger)


def create_experiment(api_key: str, project_name: str, workspasce: str) -> Experiment:
    return Experiment(
        api_key=api_key,
        project_name=project_name,
        workspace=workspasce
    )


def create_dataset(cfg: DictConfig) -> VAEDataset:
    transform = [
        v2.Resize(size=(cfg['dataset']['height'], cfg['dataset']['width'])),
    ]
    if cfg['dataset']['normalize']:
        transform.append(v2.ToDtype(torch.float32, scale=True))
    transform = v2.Compose(transform)
    
    return VAEDataset(data_path=Path(cfg['dataset']['path']), transform=transform)


def split_train_val_dataset(dataset: VAEDataset, cfg: DictConfig) -> tuple[VAEDataset, VAEDataset]:
    seed = cfg['experiment']['seed']
    dataset_split_generator = torch.Generator().manual_seed(seed)
    
    val_split = cfg['training']['train_val_split']
    train_split = 1 - val_split
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_split, val_split], dataset_split_generator)
    
    return train_dataset, val_dataset


def create_train_data_loader(train_dataset: VAEDataset, cfg: DictConfig) -> DataLoader:
    return DataLoader(train_dataset, batch_size=cfg['training']['train_batch_size'], shuffle=True)


def create_val_data_loader(val_dataset: VAEDataset, cfg: DictConfig) -> DataLoader:
    return DataLoader(val_dataset, batch_size=cfg['training']['val_batch_size'], shuffle=False)


def create_vae_model(cfg: DictConfig, device: torch.device) -> TensorDictModule:
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
    
    return TensorDictModule(vae_model, in_keys=["pixels_transformed"], out_keys=["q_z", "p_x"])


def create_vae_loss(vae_model: TensorDictModule, cfg: DictConfig, training_steps: int) -> VAELoss:
    vae_loss = VAELoss(vae_model,
                       beta=cfg['training']['kl_divergence_beta'],
                       training_steps=training_steps,
                       annealing_strategy=cfg['training']['kl_divergence_annealing_strategy'],
                       annealing_cycles=cfg['training']['kl_divergence_annealing_cycles'],
                       annealing_ratio=cfg['training']['kl_divergence_annealing_ratio'],
                       reconstruction_loss=cfg['training']['reconstruction_loss'])
    
    return vae_loss


def create_optimizer(vae_loss: VAELoss, cfg: DictConfig) -> Adam:
    if cfg['model']['optimizer']['name'] == "adam":
        optimizer = Adam(vae_loss.parameters(), lr=cfg['model']['optimizer']['lr'])
    else:
        raise ValueError(f"Unknown optimizer name: '{cfg['model']['optimizer']['name']}'")
    
    return optimizer


def log_vae_samples(vae_model: TensorDictModule, val_dataset: VAEDataset, experiment: Experiment, train_step: int, cfg: DictConfig, device: torch.device, val_prefix: str, logger: logging.Logger = logger):
    logger.info("Logging reconstructed samples...")
    
    vae_model.eval()
    
    random_eval_indxes = torch.randint(0, len(val_dataset), (cfg['logging']['val_num_preview_samples'],))
    
    random_eval_samples = []
    for random_eval_indx in random_eval_indxes:
        random_eval_samples.append(val_dataset[random_eval_indx])
    random_eval_samples = torch.stack(random_eval_samples).to(device)
    random_eval_samples = TensorDict(
        source={
            "pixels_transformed": random_eval_samples
        },
        batch_size=[random_eval_samples.shape[0]]
    )
    
    reconstructed_samples_fig = plot_vae_samples(model=vae_model, samples=random_eval_samples, loc=0., scale=1.)
    
    experiment.log_image(reconstructed_samples_fig, name=f"{val_prefix}reconstructed_samples_step_{train_step}", step=train_step)


def validate_model(vae_model: TensorDictModule, val_loader: DataLoader, device: torch.device, vae_loss: VAELoss, logger: logging.Logger, train_step: int) -> dict:
    val_losses = []
    val_losses_no_annealing = []
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
            
            loss_result = vae_loss(X_val, train_step)
            
            val_losses.append(loss_result['loss'].item())
            val_losses_no_annealing.append(loss_result['loss_no_annealing'])
            val_mean_reconstruction_losses.append(loss_result['mean_reconstruction_loss'])
            val_mean_kl_divergence_losses.append(loss_result['mean_kl_divergence_loss'])
            val_mean_kl_divergences.append(loss_result['mean_kl_divergence'])
    
    return {
        'val_losses': val_losses,
        'val_losses_no_annealing': val_losses_no_annealing,
        'val_mean_reconstruction_losses': val_mean_reconstruction_losses,
        'val_mean_kl_divergence_losses': val_mean_kl_divergence_losses,
        'val_mean_kl_divergences': val_mean_kl_divergences
    }


def train_model(vae_model: TensorDictModule, vae_loss: VAELoss, optimizer: torch.optim.Optimizer, train_loader: DataLoader, val_loader: DataLoader, cfg: DictConfig, experiment: Experiment, device: torch.device, val_dataset: VAEDataset, logger: logging.Logger, training_steps: int) -> tuple[float, Path]:
    best_val_loss = float('inf')
    best_model_path = None
    train_step = 0
    train_prefix = "train_"
    val_prefix = "val_"
    hydra_output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    val_model_save_path = hydra_output_path / Path(cfg['model']['save_dir'])
    val_model_save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting training...")
    try:
        for epoch in tqdm(range(cfg['training']['epochs'])):
            
            logger.info(f"Training Epoch {epoch}")
            train_losses = []
            train_losses_no_annealing = []
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
                
                loss_result = vae_loss(X_train, train_step)
                
                train_losses.append(loss_result['loss'].item())
                train_losses_no_annealing.append(loss_result['loss_no_annealing'])
                train_mean_reconstruction_losses.append(loss_result['mean_reconstruction_loss'])
                train_mean_kl_divergence_losses.append(loss_result['mean_kl_divergence_loss'])
                train_mean_kl_divergences.append(loss_result['mean_kl_divergence'])
                experiment.log_metric(f"{train_prefix}loss", train_losses[-1], step=train_step)
                experiment.log_metric(f"{train_prefix}loss_no_annealing", train_losses_no_annealing[-1], step=train_step)
                experiment.log_metric(f"{train_prefix}mean_reconstruction_loss", train_mean_reconstruction_losses[-1], step=train_step)
                experiment.log_metric(f"{train_prefix}mean_kl_divergence_loss", train_mean_kl_divergence_losses[-1], step=train_step)
                experiment.log_metric(f"{train_prefix}mean_kl_divergence", train_mean_kl_divergences[-1], step=train_step)
                experiment.log_metric(f"{train_prefix}kl_loss_weight", loss_result['kl_loss_weight'], step=train_step)
                
                if train_step % cfg['logging']['val_image_log_steps_interval'] == 0:
                    log_vae_samples(vae_model, val_dataset, experiment, train_step, cfg, device, val_prefix, logger)
                
                optimizer.zero_grad()
                loss_result['loss'].backward()
                optimizer.step()
                
                if train_step < training_steps - 1:
                    train_step += 1
            
            epoch_prefix = "epoch_"
            experiment.log_metric(f"{epoch_prefix}{train_prefix}loss", torch.tensor(train_losses).mean().item(), epoch=epoch)
            experiment.log_metric(f"{epoch_prefix}{train_prefix}loss_no_annealing", torch.tensor(train_losses_no_annealing).mean().item(), epoch=epoch)
            experiment.log_metric(f"{epoch_prefix}{train_prefix}mean_reconstruction_loss", torch.tensor(train_mean_reconstruction_losses).mean().item(), epoch=epoch)
            experiment.log_metric(f"{epoch_prefix}{train_prefix}mean_kl_divergence_loss", torch.tensor(train_mean_kl_divergence_losses).mean().item(), epoch=epoch)
            experiment.log_metric(f"{epoch_prefix}{train_prefix}mean_kl_divergence", torch.tensor(train_mean_kl_divergences).mean().item(), epoch=epoch)
            
            val_metrics = validate_model(vae_model, val_loader, device, vae_loss, logger, train_step)
            
            val_epoch_mean_loss_no_annealing = torch.tensor(val_metrics['val_losses_no_annealing']).mean().item()
            if val_epoch_mean_loss_no_annealing < best_val_loss:
                logger.info("Saving best model...")
                best_val_loss = val_epoch_mean_loss_no_annealing
                [f.unlink() for f in val_model_save_path.glob("*") if f.is_file()] 
                best_model_path = val_model_save_path / Path("model.pt")
                torch.save(vae_model.state_dict(), best_model_path)
            
            experiment.log_metric(f"{epoch_prefix}{val_prefix}loss", torch.tensor(val_metrics['val_losses']).mean().item(), epoch=epoch)
            experiment.log_metric(f"{epoch_prefix}{val_prefix}loss_no_annealing", val_epoch_mean_loss_no_annealing, epoch=epoch)
            experiment.log_metric(f"{epoch_prefix}{val_prefix}mean_reconstruction_loss", torch.tensor(val_metrics['val_mean_reconstruction_losses']).mean().item(), epoch=epoch)
            experiment.log_metric(f"{epoch_prefix}{val_prefix}mean_kl_divergence_loss", torch.tensor(val_metrics['val_mean_kl_divergence_losses']).mean().item(), epoch=epoch)
            experiment.log_metric(f"{epoch_prefix}{val_prefix}mean_kl_divergence", torch.tensor(val_metrics['val_mean_kl_divergences']).mean().item(), epoch=epoch)
    except InterruptedExperiment as exc:
        experiment.log_other("status", str(exc))
        logger.info("Experiment interrupted!")
    logger.info("Training done!")
    
    return best_val_loss, best_model_path


def save_model(experiment: Experiment, best_model_path: Path, logger: logging.Logger, cfg: DictConfig, vae_model: TensorDictModule):
    env_name = cfg['env']['name']
    
    if best_model_path is not None:
        logger.info("Saving best model...")
        best_model_artifact = Artifact(name=f"vae_best_model_{env_name}", artifact_type="model")
        best_model_artifact.add(best_model_path)
        experiment.log_artifact(best_model_artifact)
    else:
        logger.info("Best model is 'None', skipping saving!")
    
    if cfg['model']['save_best_model_only']:
        return
    
    logger.info("Saving model...")
    model_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / Path(cfg['model']['save_dir']) / Path("model.pt")
    if model_path.exists():
        model_path.unlink()
    torch.save(vae_model.state_dict(), model_path)
    model_artifact = Artifact(name=f"vae_model_{env_name}", artifact_type="model")
    model_artifact.add(model_path)
    experiment.log_artifact(model_artifact)


def log_log_file(experiment: Experiment):
    hydra_output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log_file_path = list(hydra_output_path.glob("*.log"))[0]
    experiment.log_asset(log_file_path)


def cleanup_resources(dataset: VAEDataset, experiment: Experiment, logger: logging.Logger):
    logger.info("Closing dataset...")
    dataset.close()
    logger.info("Ending experiment...")
    experiment.end()


if __name__ == "__main__":
    main()
