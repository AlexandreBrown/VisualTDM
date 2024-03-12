from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import hydra
import logging
import torch
import os
from omegaconf import DictConfig
from envs.env_factory import create_env
from pathlib import Path
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.data.replay_buffers import LazyTensorStorage
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.transforms import Compose
from torchrl.envs.transforms import ToTensorImage
from torchrl.envs.transforms import Resize
from torchrl.envs.transforms import ExcludeTransform
from torchrl.envs.transforms import ObservationNorm
from models.vae.asymmetrical_model import VAEModel
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import RandomPolicy
from torch.optim import Adam
from losses.vae_loss import VAELoss
from plotting.vae import plot_vae_samples


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs/", config_name="vae")
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    exp_name = cfg['experiment']['name']
    
    outputs_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    video_dir = outputs_dir / Path(cfg['logging']['video_dir'])
    train_video_dir = video_dir / Path(cfg['logging']['train_video_dir'])
    
    train_env = create_env(exp_name=exp_name,
                           seed=cfg['experiment']['seed'],
                           env_name=cfg['env']['name'],
                           video_dir=train_video_dir,
                           video_fps=cfg['logging']['video_fps'],
                           device=torch.device("cpu"),
                           resize_dim=(cfg['env']['obs']['width'], cfg['env']['obs']['height']))
    
    train_env.transform[-1].init_stats(cfg['env']['normalization_stats_init_iter'])
    
    obs_loc = train_env.transform[-1].loc
    obs_scale = train_env.transform[-1].scale
    
    policy = RandomPolicy(action_spec=train_env.action_spec)
    
    collector = SyncDataCollector(
        create_env_fn=train_env,
        policy=policy,
        total_frames=cfg['env']['train_total_frames'],
        init_random_frames=None,
        max_frames_per_traj=cfg['env']['train_max_frames_per_traj'],
        frames_per_batch=cfg['env']['train_frames_per_batch'],
        reset_at_each_iter=cfg['env']['train_reset_at_each_iter'],
        device=torch.device("cpu"),
        storing_device=torch.device("cpu"),
        postproc=ExcludeTransform("pixels_transformed", ("next", "pixels_transformed"))
    )
    
    eval_video_dir = video_dir / Path(cfg['logging']['eval_video_dir'])
    eval_env = create_env(exp_name=exp_name,
                           seed=cfg['experiment']['seed'],
                           env_name=cfg['env']['name'],
                           video_dir=eval_video_dir,
                           video_fps=cfg['logging']['video_fps'],
                           device=device,
                           resize_dim=(cfg['env']['obs']['width'], cfg['env']['obs']['height']))
    
    vae_model = VAEModel(input_dim=cfg['model']['params']['input_dim'],
                         hidden_dim=cfg['model']['params']['hidden_dim'],
                         latent_dim=cfg['model']['params']['latent_dim']).to(device)
    vae_tensordictmodule = TensorDictModule(vae_model, in_keys=["pixels_transformed"], out_keys=["q_z", "p_x"])
    
    vae_loss = VAELoss(vae_tensordictmodule, beta=cfg['alg']['kl_divergence_beta'])
    
    if cfg['model']['optimizer']['name'] == "adam":
        optimizer = Adam(vae_loss.parameters(), lr=cfg['model']['optimizer']['lr'])
    else:
        raise ValueError(f"Unknown optimizer name: '{cfg['model']['optimizer']['name']}'")
    
    replay_buffer_transform = Compose(
        ToTensorImage(
            in_keys=["pixels", ("next", "pixels")],
            out_keys=["pixels_transformed", ("next", "pixels_transformed")],
        ),
        Resize(in_keys=["pixels_transformed", ("next", "pixels_transformed")], w=cfg['env']['obs']['width'], h=cfg['env']['obs']['height']),
        ObservationNorm(loc=obs_loc, scale=obs_scale, in_keys=["pixels_transformed"], out_keys=["pixels_transformed"], standard_normal=True)
    )
    
    buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=cfg['replay_buffer']['max_size']), transform=replay_buffer_transform)
    
    step = 0
    
    with experiment.train():
        for data in collector:
            buffer.extend(data)
            for _ in range(cfg['alg']['optim_steps_per_iter']):
                vae_tensordictmodule.train()
                train_batch = buffer.sample(cfg['alg']['train_batch_size']).to(device)
                loss_result = vae_loss(train_batch)
                
                logger.info(f"Loss: {loss_result['loss'].item()}")
                experiment.log_metric("loss", loss_result['loss'].item(), step=step)
                experiment.log_metric("mean_log_p_x_given_z", loss_result['mean_log_p_x_given_z'], step=step)
                experiment.log_metric("mean_kl_divergence_q_z", loss_result['mean_kl_divergence_q_z'], step=step)
                
                optimizer.zero_grad()
                loss_result['loss'].backward()
                optimizer.step()
                
                if step % cfg['logging']['train_plot_interval'] == 0:
                    logger.info("Generating and logging reconstructed samples...")
                    reconstructed_samples_img = plot_vae_samples(model=vae_tensordictmodule, x=train_batch, num_samples=cfg['logging']['train_plot_samples'], loc=obs_loc, scale=obs_scale)
                    experiment.log_image(reconstructed_samples_img, name=f"reconstructed_samples_step_{step}", step=step)
                
                step += 1
    
    log_model(experiment, vae_model, model_name="vae_model")
    
    train_env.transform.dump()
    train_env.close()
    
    #eval_env.transform.dump()
    eval_env.close()

if __name__ == "__main__":
    main()
