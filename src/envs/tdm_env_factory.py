from omegaconf import DictConfig
from tensordict.nn import TensorDictModule
from torchrl.envs import TransformedEnv
from envs.transforms.add_obs_latent_representation import AddObsLatentRepresentation
from envs.transforms.add_goal_reached import AddGoalReached
from envs.transforms.add_goal_vector_distance_reward import AddGoalVectorDistanceReward
from envs.transforms.add_planning_horizon import AddPlanningHorizon
from envs.transforms.add_goal_latent_representation import AddGoalLatentRepresentation
from envs.env_factory import create_env
from envs.max_planning_horizon_scheduler import MaxPlanningHorizonScheduler
from torchrl.envs.transforms import DoubleToFloat
from torchrl.envs.transforms import CatTensors


def create_tdm_env(cfg: DictConfig, encoder: TensorDictModule, max_planning_horizon_scheduler: MaxPlanningHorizonScheduler) -> TransformedEnv:
    env = create_env(env_name=cfg['env']['name'],
                           seed=cfg['experiment']['seed'],
                           device=cfg['env']['device'],
                           normalize_obs=cfg['env']['obs']['normalize'],
                           standardization_stats_init_iter=cfg['env']['obs']['standardization_stats_init_iter'],
                           standardize_obs=cfg['env']['obs']['standardize'],
                           raw_height=cfg['env']['obs']['raw_height'],
                           raw_width=cfg['env']['obs']['raw_width'],
                           resize_dim=(cfg['env']['obs']['width'], cfg['env']['obs']['height']))
    
    env.append_transform(AddPlanningHorizon(max_planning_horizon_scheduler=max_planning_horizon_scheduler))
    
    env.append_transform(AddGoalLatentRepresentation(encoder_decoder_model=encoder,
                                                           latent_dim=cfg['env']['goal']['latent_dim']))
    
    env.append_transform(AddObsLatentRepresentation(encoder=encoder,
                                                          latent_dim=cfg['env']['goal']['latent_dim']))
    env.append_transform(AddGoalReached(goal_reached_epsilon=cfg['env']['goal']['reached_epsilon']))
    
    env.append_transform(DoubleToFloat(in_keys=['observation'], out_keys=['state']))
    
    env.append_transform(DoubleToFloat(in_keys=['desired_goal'], out_keys=['desired_goal']))
    
    env.append_transform(AddGoalVectorDistanceReward(norm_type=cfg['train']['reward_norm_type'],
                                                           latent_dim=cfg['env']['goal']['latent_dim']))
    
    env.append_transform(CatTensors(in_keys=list(cfg['models']['actor']['in_keys']), out_key="actor_inputs", del_keys=False))
    
    return env
