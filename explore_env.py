import hydra
import torch
import logging
from omegaconf import DictConfig
from pathlib import Path
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.collectors.collectors import RandomPolicy
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record import VideoRecorder
from tqdm import tqdm
from envs.env_factory import create_env


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs/", config_name="env_exploration")
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
    
    video_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / Path(cfg['logging']['video_dir'])
    video_logger = CSVLogger(exp_name=cfg['experiment']['name'], log_dir=video_dir, video_format="mp4", video_fps=cfg['logging']['video_fps'])
    recorder = VideoRecorder(logger=video_logger, tag="iteration")
    
    env.append_transform(recorder)
    
    policy = RandomPolicy(action_spec=env.action_spec)
    
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=policy,
        total_frames=cfg['exploration']['total_frames'],
        max_frames_per_traj=cfg['exploration']['max_frames_per_traj'],
        frames_per_batch=cfg['exploration']['frames_per_batch'],
        reset_at_each_iter=cfg['exploration']['reset_at_each_iter'],
        device=device,
        storing_device=device,
    )

    logger.info("Exploring env...")
    
    for _ in tqdm(collector):
        continue
    
    logger.info("Done exploring env!")
    
    logger.info("Saving video...")
    env.transform.dump()
    logger.info("Closing env...")
    env.close()

if __name__ == "__main__":
    main()
