import hydra
import torch
import logging
import h5py
from omegaconf import DictConfig
from pathlib import Path
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.collectors.collectors import RandomPolicy
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record import VideoRecorder
from tensordict import TensorDict
from tqdm import tqdm
from envs.env_factory import create_env


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs/", config_name="vae_dataset_generation")
def main(cfg: DictConfig):
    device = torch.device("cpu")
    
    env = create_env(env_name=cfg['env']['name'],
                     seed=cfg['experiment']['seed'],
                     device=device,
                     normalize_obs=False,
                     standardization_stats_init_iter=0,
                     standardize_obs=False,
                     raw_height=cfg['env']['obs']['raw_height'],
                     raw_width=cfg['env']['obs']['raw_width'],
                     resize_dim=None)
    
    hydra_output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    video_dir = hydra_output_path / Path(cfg['logging']['video_dir'])
    video_logger = CSVLogger(exp_name=cfg['experiment']['name'], log_dir=video_dir, video_format="mp4", video_fps=cfg['logging']['video_fps'])
    recorder = VideoRecorder(logger=video_logger, tag="iteration", make_grid=False)
    
    dataset_save_path = hydra_output_path / Path(cfg['dataset']['save_dir']) / Path(cfg['dataset']['name'] + ".h5")
    dataset_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    policy = RandomPolicy(action_spec=env.action_spec)
    
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=policy,
        total_frames=cfg['env']['total_frames'],
        max_frames_per_traj=cfg['env']['max_frames_per_traj'],
        frames_per_batch=cfg['env']['frames_per_batch'],
        reset_at_each_iter=cfg['env']['reset_at_each_iter'],
        device=device,
        storing_device=device,
    )

    logger.info("Collecting data from env...")
    
    datafile = h5py.File(dataset_save_path, 'w')
    env_obs_shape = env.observation_spec._specs['pixels'].shape
    datafile.create_dataset(dataset_save_path.stem, (cfg['env']['total_frames'],env_obs_shape[0],env_obs_shape[1],env_obs_shape[2]), dtype='uint8')
    for i, data in enumerate(tqdm(collector)):
        save_data(data, datafile, dataset_save_path.stem, i)
        log_video(env, recorder, i, cfg)
    datafile.close()
    logger.info("Done collecting data!")
    
    logger.info("Closing env...")
    env.close()


def save_data(data: TensorDict, datafile, dataset_name, i):
    images = data['pixels'].numpy()
    
    logger.info("Updating dataset with new data...")
    
    batch_len = images.shape[0]
    datafile[dataset_name][i*batch_len:i*batch_len + batch_len, :, :, :] = images

def log_video(env, recorder: VideoRecorder, i: int ,cfg):
    if i * cfg['env']['frames_per_batch'] % cfg['logging']['video_log_steps_interval'] != 0:
        return
    
    logger.info("Collecting data for video...")
    for _ in tqdm(range(cfg['logging']['video_rollouts'])):
        video_obs = env.rollout(max_steps=cfg['logging']['video_max_steps'])
        for obs in video_obs['pixels']:
            recorder._apply_transform(obs)
    
    logger.info("Saving video...")
    recorder.dump()

if __name__ == "__main__":
    main()
