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
from envs.goal_env_factory import create_env


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs/", config_name="env_exploration")
def main(cfg: DictConfig):
    env = create_env(cfg=cfg,
                     normalize_obs=False,
                     standardization_stats_init_iter=0,
                     standardize_obs=False,
                     resize_width_height=None)
    
    video_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / Path(cfg['logging']['video_dir'])
    video_logger = CSVLogger(exp_name=cfg['experiment']['name'], log_dir=video_dir, video_format="mp4", video_fps=cfg['logging']['video_fps'])
    recorder = VideoRecorder(logger=video_logger, tag="iteration", skip=1)
    
    env.append_transform(recorder)
    
    policy = RandomPolicy(action_spec=env.action_spec)
    
    device = torch.device(cfg['env']['device'])
    
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

    logger.info("Exploring env...")
    
    for data in tqdm(collector):
        continue
    
    logger.info("Done exploring env!")
    
    logger.info("Saving video...")
    env.transform.dump()
    logger.info("Closing env...")
    env.close()

if __name__ == "__main__":
    main()
