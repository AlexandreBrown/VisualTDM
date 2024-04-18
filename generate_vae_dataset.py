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
from tqdm import tqdm
from envs.env_factory import create_env
from plotting.image import get_images_grid


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
                     resize_dim=None,
                     goal_x_min_max=list(cfg['env']['goal']['x_min_max']),
                     goal_y_min_max=list(cfg['env']['goal']['y_min_max']),
                     goal_z_min_max=list(cfg['env']['goal']['z_min_max']),
                     camera_distance=cfg['env']['camera']['distance'])
    
    hydra_output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    video_dir = hydra_output_path / Path(cfg['logging']['video_dir'])
    video_logger = CSVLogger(exp_name=cfg['experiment']['name'], log_dir=video_dir, video_format="mp4", video_fps=cfg['logging']['video_fps'])
    recorder = VideoRecorder(logger=video_logger, tag="iteration", make_grid=False, skip=1)
    
    images_dir = hydra_output_path / Path(cfg['logging']['images_dir'])
    images_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_save_path = hydra_output_path / Path(cfg['dataset']['save_dir']) / Path(cfg['dataset']['name'] + ".h5")
    dataset_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    policy = RandomPolicy(action_spec=env.action_spec)
    
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=policy,
        total_frames=cfg['dataset']['size'] // 2, # Because we save pixels + goal_pixels every iter so each iter = 2 frames
        max_frames_per_traj=cfg['env']['max_frames_per_traj'],
        frames_per_batch=cfg['env']['frames_per_batch'],
        reset_at_each_iter=cfg['env']['reset_at_each_iter'],
        device=device,
        storing_device=device,
    )

    logger.info("Collecting data from env...")
    
    datafile = h5py.File(dataset_save_path, 'w')
    env_obs_shape = env.observation_spec._specs['pixels'].shape
    datafile.create_dataset(dataset_save_path.stem, (cfg['dataset']['size'],env_obs_shape[0],env_obs_shape[1],env_obs_shape[2]), dtype='uint8')
    for i, data in enumerate(tqdm(collector)):
        save_data(data['pixels'], datafile, dataset_save_path.stem, i)
        save_data(data['goal_pixels'], datafile, dataset_save_path.stem, i+1)
        log_video(env, recorder, i, cfg)
        log_images(datafile, dataset_save_path.stem, data, i, cfg, images_dir)
    datafile.close()
    logger.info("Done collecting data!")
    
    logger.info("Closing env...")
    env.close()


def save_data(data: torch.Tensor, datafile, dataset_name, i):
    data = data.numpy()
    
    logger.info("Updating dataset with new data...")
    
    batch_len = data.shape[0]
    datafile[dataset_name][i*batch_len:i*batch_len + batch_len, :, :, :] = data

def log_video(env, recorder: VideoRecorder, i: int ,cfg):
    if i * cfg['env']['frames_per_batch'] % cfg['logging']['video_log_steps_interval'] != 0:
        return
    
    logger.info("Collecting data for video...")
    for _ in tqdm(range(cfg['logging']['video_rollouts'])):
        video_obs = env.rollout(max_steps=cfg['logging']['video_max_frames'])
        for obs in video_obs['pixels']:
            recorder._apply_transform(obs)
    
    logger.info("Saving video...")
    recorder.dump()


def log_images(datafile, dataset_name, data: torch.Tensor, i: int, cfg: DictConfig, images_dir):
    if i * cfg['env']['frames_per_batch'] % cfg['logging']['images_log_steps_interval'] != 0:
        return

    logger.info("Saving images...")
    
    batch_len = data.shape[0]
    images_grid_rows = cfg['logging']['images_grid_rows']
    images_grid_columns = cfg['logging']['images_grid_columns']
    total_images_to_log = images_grid_rows * images_grid_columns
    
    pixels_low_index = i*batch_len
    pixels_high_index = pixels_low_index + batch_len
    
    goal_pixels_low_index = (i+1)*batch_len
    goal_pixels_high_index = goal_pixels_low_index + batch_len
    
    random_images_idx_1 = torch.randint(low=pixels_low_index, high=pixels_high_index, size=(total_images_to_log//2,))
    random_images_idx_2 = torch.randint(low=goal_pixels_low_index, high=goal_pixels_high_index, size=(total_images_to_log//2,))
    random_images_idx = torch.cat([random_images_idx_1, random_images_idx_2], dim=0).sort(descending=False)[0]
    
    images = []
    for random_image_idx in random_images_idx:
        image = datafile[dataset_name][random_image_idx]
        image = torch.from_numpy(image).permute(2,0,1)
        images.append(image)
    images = torch.stack(images, dim=0)
    
    images_fig = get_images_grid(images, num_rows=images_grid_rows, num_cols=images_grid_columns)
    images_fig.savefig(Path(images_dir) / Path(f"iteration_{i}.png"))
if __name__ == "__main__":
    main()
