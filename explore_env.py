import hydra
import torch
from omegaconf import DictConfig
from pathlib import Path
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.collectors.collectors import RandomPolicy
from tqdm import tqdm
from envs.env_factory import create_env


@hydra.main(version_base=None, config_path="configs/", config_name="env_exploration")
def main(cfg: DictConfig):

    video_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / Path(cfg['logging']['video_dir'])
    
    device = torch.device("cpu")
    
    env = create_env(exp_name=cfg['experiment']['name'],
                     seed=cfg['experiment']['seed'],
                     env_name=cfg['env']['name'],
                     video_dir=str(video_dir.absolute()),
                     video_fps=cfg['logging']['video_fps'],
                     device=device,
                     resize_dim=None)
    
    policy = RandomPolicy(action_spec=env.action_spec)
    
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=policy,
        total_frames=400,
        max_frames_per_traj=100,
        frames_per_batch=200,
        reset_at_each_iter=False,
        device=device,
        storing_device=device,
    )

    print("Exploring env...")
    
    for _ in tqdm(collector):
        continue
    
    print("Done exploring env!")
    
    print("Closing env...")
    env.transform.dump()
    env.close()

if __name__ == "__main__":
    main()
