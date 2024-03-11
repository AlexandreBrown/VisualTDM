import hydra
from omegaconf import DictConfig
from envs.exploration import explore_env
from pathlib import Path


@hydra.main(version_base=None, config_path="configs/", config_name="env_exploration")
def main(cfg: DictConfig):
    exp_name = cfg['experiment']['name']
    seed = cfg['experiment']['seed']
    env_name = cfg['env']['name']
    video_fps = cfg['logging']['video_fps']
    video_dir = cfg['logging']['video_dir']
    
    video_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / Path(video_dir)
    
    explore_env(exp_name=exp_name,
                seed=seed,
                env_name=env_name,
                video_dir=str(video_dir.absolute()),
                video_fps=video_fps)

if __name__ == "__main__":
    main()
