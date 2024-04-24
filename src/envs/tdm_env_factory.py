import torch
from omegaconf import DictConfig
from tensordict.nn import TensorDictModule
from torchrl.envs import TransformedEnv
from envs.max_planning_horizon_scheduler import TdmMaxPlanningHorizonScheduler
from envs.transforms.add_obs_latent_representation import AddObsLatentRepresentation
from envs.transforms.add_goal_vector_distance_reward import AddGoalVectorDistanceReward
from envs.transforms.add_step_planning_horizon import AddStepPlanningHorizon
from envs.transforms.add_goal_latent_representation import AddGoalLatentRepresentation
from envs.goal_env_factory import create_env
from torchrl.envs.transforms import DoubleToFloat
from torchrl.envs.transforms import CatTensors
from torchrl.envs.transforms import RenameTransform
from torchrl.envs.transforms import ObservationNorm
from envs.transforms.add_tdm_done import AddTdmDone


def create_tdm_env(cfg: DictConfig, encoder: TensorDictModule, tdm_max_planning_horizon_scheduler: TdmMaxPlanningHorizonScheduler, goal_loc: torch.Tensor = None, goal_scale: torch.Tensor = None, state_loc: torch.Tensor = None, state_scale: torch.Tensor = None) -> TransformedEnv:
    env = create_env(cfg=cfg,
                     normalize_obs=cfg['env']['obs']['normalize'],
                     standardization_stats_init_iter=cfg['env']['obs']['standardization_stats_init_iter'],
                     standardize_obs=cfg['env']['obs']['standardize'],
                     resize_width_height=(cfg['env']['obs']['width'], cfg['env']['obs']['height']))
    
    env.append_transform(RenameTransform(in_keys=['reward'], out_keys=['original_reward'], create_copy=True))
    
    env.append_transform(AddStepPlanningHorizon(tdm_rollout_max_planning_horizon=cfg['train']['tdm_rollout_max_planning_horizon'],
                                                tdm_max_planning_horizon_scheduler=tdm_max_planning_horizon_scheduler))
    
    env.append_transform(AddGoalLatentRepresentation(encoder_decoder_model=encoder,
                                                     latent_dim=cfg['env']['goal']['latent_dim']))
    
    env.append_transform(AddObsLatentRepresentation(encoder=encoder,
                                                    latent_dim=cfg['env']['goal']['latent_dim']))
    
    goal_norm_transform = None
    state_norm_transform = None
    
    if "observation" in cfg['env']['keys_of_interest'] and "state" in cfg['env']['keys_of_interest']:
        env.append_transform(DoubleToFloat(in_keys=['observation'], out_keys=['state']))
        if cfg['env']['state']['normalize']:
            if state_loc is None or state_scale is None:
                state_norm_transform = ObservationNorm(in_keys=['state'], out_keys=['state'], standard_normal=cfg['env']['state']['standardize'])
                env.append_transform(state_norm_transform)
                state_norm_transform.init_stats(num_iter=4096)
            else:
                env.append_transform(ObservationNorm(in_keys=['state'], out_keys=['state'], loc=state_loc, scale=state_scale, standard_normal=cfg['env']['state']['standardize']))

    if 'achieved_goal' in cfg['env']['keys_of_interest']:
        env.append_transform(DoubleToFloat(in_keys=['achieved_goal'], out_keys=['achieved_goal']))
        if cfg['env']['goal']['normalize']:
            if goal_loc is None or goal_scale is None:
                goal_norm_transform = ObservationNorm(in_keys=['achieved_goal'], out_keys=['achieved_goal'], standard_normal=cfg['env']['goal']['standardize'])
                env.append_transform(goal_norm_transform)
                goal_norm_transform.init_stats(num_iter=4096)
            else:
                env.append_transform(ObservationNorm(in_keys=['achieved_goal'], out_keys=['achieved_goal'], loc=goal_loc, scale=goal_scale, standard_normal=cfg['env']['goal']['standardize']))

    if 'desired_goal' in cfg['env']['keys_of_interest']:
        env.append_transform(DoubleToFloat(in_keys=['desired_goal'], out_keys=['desired_goal']))
        if cfg['env']['goal']['normalize']:
            if goal_loc is None or goal_scale is None:
                env.append_transform(ObservationNorm(in_keys=['desired_goal'], out_keys=['desired_goal'], loc=goal_norm_transform.loc, scale=goal_norm_transform.scale, standard_normal=cfg['env']['goal']['standardize']))
            else:
                env.append_transform(ObservationNorm(in_keys=['desired_goal'], out_keys=['desired_goal'], loc=goal_loc, scale=goal_scale, standard_normal=cfg['env']['goal']['standardize']))
    
    env.append_transform(AddGoalVectorDistanceReward(distance_type=cfg['train']['reward_distance_type'],
                                                     reward_dim=cfg['train']['reward_dim']))
    
    env.append_transform(AddTdmDone(max_frames_per_traj=cfg['env']['max_frames_per_traj'], terminate_when_goal_reached=cfg['train']['tdm_terminate_when_goal_reached'], goal_reached_epsilon=cfg['env']['goal']['reached_epsilon']))
    
    env.append_transform(CatTensors(in_keys=list(cfg['models']['actor']['in_keys']), out_key="actor_inputs", del_keys=False))
    
    return env, goal_norm_transform, state_norm_transform
