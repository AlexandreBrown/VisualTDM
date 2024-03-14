from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from comet_ml.exceptions import InterruptedExperiment
import hydra
import logging
import torch
import os
from omegaconf import DictConfig
from envs.env_factory import create_env
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.data.replay_buffers import LazyMemmapStorage
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.transforms import Compose
from torchrl.envs.transforms import ToTensorImage
from torchrl.envs.transforms import Resize
from torchrl.envs.transforms import ExcludeTransform
from torchrl.envs.transforms import ObservationNorm
from models.vae.model import VAEModel
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import RandomPolicy
from torch.optim import Adam
from losses.vae_loss import VAELoss
from plotting.vae import plot_vae_samples


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
  
    
    policy = RandomPolicy(action_spec=train_env.action_spec)
    
    collector = SyncDataCollector(
        create_env_fn=train_env,
        policy=policy,
        total_frames=-1,
        init_random_frames=None,
        max_frames_per_traj=cfg['env']['train_max_frames_per_traj'],
        frames_per_batch=cfg['env']['train_frames_per_batch'],
        reset_at_each_iter=cfg['env']['train_reset_at_each_iter'],
        device=torch.device("cpu"),
        storing_device=torch.device("cpu"),
        postproc=ExcludeTransform("pixels_transformed", ("next", "pixels_transformed"))
    )
    
    encoder_params = cfg['model']['encoder']
    decoder_params = cfg['model']['decoder']
    vae_model = VAEModel(input_spatial_dim=cfg['env']['obs']['height'],
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
    vae_tensordictmodule = TensorDictModule(vae_model, in_keys=["pixels_transformed"], out_keys=["q_z", "p_x"])
    
    vae_loss = VAELoss(vae_tensordictmodule,
                       beta=cfg['alg']['kl_divergence_beta'],
                       training_steps=cfg['alg']['max_train_steps'],
                       annealing_strategy=cfg['alg']['kl_divergence_annealing_strategy'],
                       annealing_cycles=cfg['alg']['kl_divergence_annealing_cycles'],
                       annealing_ratio=cfg['alg']['kl_divergence_annealing_ratio'],
                       reconstruction_loss=cfg['alg']['reconstruction_loss'])
    
    if cfg['model']['optimizer']['name'] == "adam":
        optimizer = Adam(vae_loss.parameters(), lr=cfg['model']['optimizer']['lr'])
    else:
        raise ValueError(f"Unknown optimizer name: '{cfg['model']['optimizer']['name']}'")
    
    rb_transforms = []
    if cfg['env']['obs']['normalize']:
        rb_transforms.append(
            ToTensorImage(
                in_keys=["pixels", ("next", "pixels")],
                out_keys=["pixels_transformed", ("next", "pixels_transformed")],
            )
        )
    rb_transforms.append(Resize(in_keys=["pixels_transformed", ("next", "pixels_transformed")], w=cfg['env']['obs']['width'], h=cfg['env']['obs']['height']))
    
    obs_loc = 0.
    obs_scale = 1.
    
    if cfg['env']['obs']['standardize']:
        obs_loc = train_env.transform[-1].loc
        obs_scale = train_env.transform[-1].scale
        rb_transforms.append(ObservationNorm(loc=obs_loc, scale=obs_scale, in_keys=["pixels_transformed"], out_keys=["pixels_transformed"], standard_normal=True))
    
    replay_buffer_transform = Compose(*rb_transforms)
    
    buffer = ReplayBuffer(storage=LazyMemmapStorage(max_size=cfg['replay_buffer']['max_size']), transform=replay_buffer_transform)
    
    try:
        train(buffer, experiment, collector, cfg, vae_tensordictmodule, obs_loc, obs_scale, optimizer, device, vae_loss)
    except InterruptedExperiment as exc:
        experiment.log_other("status", str(exc))
        logger.info("Experiment interrupted!")
    
    log_model(experiment, vae_tensordictmodule, model_name="vae_model")
    
    train_env.close()


def train(buffer, experiment, collector, cfg, vae_tensordictmodule, obs_loc, obs_scale, optimizer, device, vae_loss: VAELoss):
    training_step_counter = 0
    
    with experiment.train():
        for data in collector:
            buffer.extend(data)
            for _ in range(cfg['alg']['train_steps_per_iter']):
                
                if training_step_counter == cfg['alg']['max_train_steps']:
                    return
                
                vae_tensordictmodule.train()
                train_batch = buffer.sample(cfg['alg']['train_batch_size']).to(device)
                loss_result = vae_loss(train_batch)
                
                experiment.log_metric("loss", loss_result['loss'].item(), step=training_step_counter)
                experiment.log_metric("mean_reconstruction_loss", loss_result['mean_reconstruction_loss'], step=training_step_counter)
                experiment.log_metric("mean_kl_divergence_loss", loss_result['mean_kl_divergence_loss'], step=training_step_counter)
                experiment.log_metric("mean_kl_divergence", loss_result['mean_kl_divergence'], step=training_step_counter)
                experiment.log_metric("kl_loss_weight", loss_result['kl_loss_weight'], step=training_step_counter)
                
                if training_step_counter % cfg['logging']['log_steps_interval'] == 0:
                    logger.info("Generating and logging reconstructed samples...")
                    reconstructed_samples_fig = plot_vae_samples(model=vae_tensordictmodule, x=train_batch, num_samples=cfg['logging']['train_plot_samples'], loc=obs_loc, scale=obs_scale)
                    experiment.log_image(reconstructed_samples_fig, name=f"reconstructed_samples_step_{training_step_counter}", step=training_step_counter)
                
                optimizer.zero_grad()
                loss_result['loss'].backward()
                optimizer.step()
                
                training_step_counter += 1


if __name__ == "__main__":
    main()
