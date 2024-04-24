import hydra
import torch
import logging
import h5py
from omegaconf import DictConfig
from pathlib import Path
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.envs.utils import RandomPolicy
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record import VideoRecorder
from tqdm import tqdm
from envs.goal_env_factory import create_env
from plotting.image import get_images_grid


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs/", config_name="vae_dataset_generation")
def main(cfg: DictConfig):
    device = torch.device("cpu")
    
    env = create_env(cfg=cfg,
                     normalize_obs=False,
                     standardization_stats_init_iter=0,
                     standardize_obs=False,
                     resize_width_height=None)
    
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
        total_frames=cfg['dataset']['size'],
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
        batch_size = data.shape[0]
        
        traj_ids = data['collector']['traj_ids']
        
        last_id = None
        unique_indexes = []
        for index, id in enumerate(traj_ids):
            if last_id is None or id != last_id:
                unique_indexes.append(index)
            
            last_id = id
        
        unique_goal_pixels = data['goal_pixels'][unique_indexes]
        
        nb_obs = batch_size - len(unique_goal_pixels)
        
        pixels_start_index = i*batch_size
        pixels_end_index = i*batch_size + nb_obs
        save_data(data['pixels'][:nb_obs], datafile, dataset_save_path.stem, ds_start_index=pixels_start_index, ds_end_index=pixels_end_index)
        
        nb_goals = batch_size - nb_obs
        goals_start_index = pixels_end_index
        goals_end_index = goals_start_index + nb_goals
        save_data(unique_goal_pixels, datafile, dataset_save_path.stem, ds_start_index=goals_start_index, ds_end_index=goals_end_index)
        
        log_video(env, recorder, i, cfg)
        log_images(datafile, dataset_save_path.stem, pixels_start_index, pixels_end_index, goals_start_index, goals_end_index, i, cfg, images_dir)
    datafile.close()
    logger.info("Done collecting data!")
    
    logger.info("Closing env...")
    env.close()


def save_data(data: torch.Tensor, datafile, dataset_name, ds_start_index: int, ds_end_index: int):
    data = data.numpy()
    logger.info("Updating dataset with new data...")
    datafile[dataset_name][ds_start_index:ds_end_index, :, :, :] = data

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


def log_images(datafile, dataset_name, pixels_start_index, pixels_end_index, goals_start_index, goals_end_index, i: int, cfg: DictConfig, images_dir):
    if i * cfg['env']['frames_per_batch'] % cfg['logging']['images_log_steps_interval'] != 0:
        return

    logger.info("Saving images...")
    
    images_grid_rows = cfg['logging']['images_grid_rows']
    images_grid_columns = cfg['logging']['images_grid_columns']
    total_images_to_log = images_grid_rows * images_grid_columns

    nb_obs = total_images_to_log//2
    nb_goals = total_images_to_log - nb_obs
    
    random_images_idx_1 = torch.randint(low=pixels_start_index, high=pixels_end_index, size=(nb_obs,))
    random_images_idx_2 = torch.randint(low=goals_start_index, high=goals_end_index, size=(nb_goals,))
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
