from omegaconf import DictConfig
from tensordict.nn import TensorDictModule
from torchrl.envs import TransformedEnv
from envs.max_planning_horizon_scheduler import TdmMaxPlanningHorizonScheduler
from envs.transforms.add_obs_latent_representation import AddObsLatentRepresentation
from envs.transforms.add_goal_reached import AddGoalLatentReached
from envs.transforms.add_goal_vector_distance_reward import AddGoalVectorDistanceReward
from envs.transforms.add_step_planning_horizon import AddStepPlanningHorizon
from envs.transforms.add_goal_latent_representation import AddGoalLatentRepresentation
from envs.env_factory import create_env
from torchrl.envs.transforms import DoubleToFloat
from torchrl.envs.transforms import CatTensors


def create_tdm_env(cfg: DictConfig, encoder: TensorDictModule, tdm_max_planning_horizon_scheduler: TdmMaxPlanningHorizonScheduler) -> TransformedEnv:
    env = create_env(cfg=cfg,
                     normalize_obs=cfg['env']['obs']['normalize'],
                     standardization_stats_init_iter=cfg['env']['obs']['standardization_stats_init_iter'],
                     standardize_obs=cfg['env']['obs']['standardize'],
                     resize_width_height=(cfg['env']['obs']['width'], cfg['env']['obs']['height']))
    
    env.append_transform(AddStepPlanningHorizon(tdm_rollout_max_planning_horizon=cfg['train']['tdm_rollout_max_planning_horizon'],
                                                tdm_max_planning_horizon_scheduler=tdm_max_planning_horizon_scheduler))
    
    env.append_transform(AddGoalLatentRepresentation(encoder_decoder_model=encoder,
                                                     latent_dim=cfg['env']['goal']['latent_dim']))
    
    env.append_transform(AddObsLatentRepresentation(encoder=encoder,
                                                    latent_dim=cfg['env']['goal']['latent_dim']))
    
    env.append_transform(AddGoalLatentReached(goal_reached_epsilon=cfg['env']['goal']['reached_epsilon'],
                                              terminate_when_goal_reached=cfg['train']['tdm_terminate_when_goal_reached'],
                                              distance_type=cfg['train']['reward_distance_type']))
    
    if "observation" in cfg['env']['keys_of_interest'] \
        and "state" in cfg['env']['keys_of_interest']:
        env.append_transform(DoubleToFloat(in_keys=['observation'], out_keys=['state']))
    
    if "desired_goal" in cfg['env']['keys_of_interest']:
        env.append_transform(DoubleToFloat(in_keys=['desired_goal'], out_keys=['desired_goal']))
    
    env.append_transform(AddGoalVectorDistanceReward(distance_type=cfg['train']['reward_distance_type'],
                                                     reward_dim=cfg['train']['reward_dim']))
    
    env.append_transform(CatTensors(in_keys=list(cfg['models']['actor']['in_keys']), out_key="actor_inputs", del_keys=False))
    
    return env
