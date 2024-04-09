from comet_ml import Experiment, OfflineExperiment
from comet_ml.exceptions import InterruptedExperiment
import hydra
import logging
import torch
import os
from omegaconf import DictConfig
from envs.env_factory import create_env
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.collectors.collectors import DataCollectorBase
from torchrl.data.replay_buffers import LazyMemmapStorage
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.transforms import Compose
from torchrl.envs.transforms import ToTensorImage
from torchrl.envs.transforms import Resize
from torchrl.envs.transforms import ExcludeTransform
from torchrl.envs.transforms import ObservationNorm
from torchrl.envs.utils import RandomPolicy
from torchrl.modules.tensordict_module import OrnsteinUhlenbeckProcessWrapper
from torchrl.envs.transforms import StepCounter
from envs.transforms.step_planning_horizon import StepPlanningHorizon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@hydra.main(version_base=None, config_path="configs/", config_name="tdm_training")
def main(cfg: DictConfig):
    COMET_ML_API_KEY = os.getenv("COMET_ML_API_KEY")
    COMET_ML_PROJECT_NAME = os.getenv("COMET_ML_PROJECT_NAME")
    COMET_ML_WORKSPACE = os.getenv("COMET_ML_WORKSPACE")
    
    experiment = create_experiment(api_key=COMET_ML_API_KEY, project_name=COMET_ML_PROJECT_NAME, workspasce=COMET_ML_WORKSPACE)
    
    experiment.log_parameters(cfg)
    experiment.log_code(folder='src')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    train_env = create_env(env_name=cfg['env']['name'],
                           seed=cfg['experiment']['seed'],
                           device=device,
                           normalize_obs=cfg['env']['obs']['normalize'],
                           standardization_stats_init_iter=cfg['env']['obs']['standardization_stats_init_iter'],
                           standardize_obs=cfg['env']['obs']['standardize'],
                           resize_dim=(cfg['env']['obs']['width'], cfg['env']['obs']['height']))
    
    policy = RandomPolicy(action_spec=train_env.action_spec)
    exploration_policy = OrnsteinUhlenbeckProcessWrapper(policy=policy)
    
    train_collector = SyncDataCollector(
        create_env_fn=train_env,
        policy=exploration_policy,
        total_frames=-1,
        init_random_frames=cfg['train']['init_random_frames'],
        max_frames_per_traj=cfg['train']['max_frames_per_traj'],
        frames_per_batch=cfg['train']['frames_per_batch'],
        reset_at_each_iter=cfg['train']['reset_at_each_iter'],
        device=torch.device(cfg['train']['collector_device']),
        storing_device=torch.device(cfg['train']['storing_device']),
        postproc=ExcludeTransform("pixels_transformed", ("next", "pixels_transformed"))
    )
    
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
    
    rb = ReplayBuffer(storage=LazyMemmapStorage(max_size=cfg['replay_buffer']['max_size']), transform=replay_buffer_transform)
    
    train_env.append_transform(StepPlanningHorizon(out_keys="planning_horizon", max_planning_horizon=cfg['train']['max_planning_horizon']))
    
    try:
        train(experiment, train_collector, rb, num_episodes=cfg['train']['num_episodes'])          
    except InterruptedExperiment as exc:
        experiment.log_other("status", str(exc))
        logger.info("Experiment interrupted!")
    
    train_env.close()


def train(experiment: Experiment, train_collector: DataCollectorBase, rb: ReplayBuffer, num_episodes: int):
    with experiment.train():
        for n in range(num_episodes):
            for t, data in enumerate(train_collector):
                action = 0 # TODO get action from DDPG policy
                    
                    


def create_experiment(api_key: str, project_name: str, workspasce: str) -> Experiment:
    return OfflineExperiment(
        api_key=api_key,
        project_name=project_name,
        workspace=workspasce
    )


if __name__ == "__main__":
    main()
