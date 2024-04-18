import logging
import gymnasium as gym
import torch
import numpy as np
import random
from torchrl.envs.transforms import TransformedEnv
from torchrl.envs.transforms import Compose
from torchrl.envs.transforms import ToTensorImage
from torchrl.envs.transforms import Resize
from torchrl.envs.transforms import ObservationNorm
from torchrl.envs import GymWrapper
from typing import Optional
from envs.gym_env_goal_strategy import AntMazeEnvGoalStrategy, FrankaKitchenEnvGoalStrategy, PointMazeEnvGoalStrategy, AndroitHandRelocateEnvGoalStrategy
from envs.goal_env import GoalEnv
from torchrl.envs.utils import check_env_specs
from envs.transforms.remove_data_from_observation import RemoveDataFromObservation

logger = logging.getLogger(__name__)

def create_env(env_name: str,
               seed: int,
               device: torch.device,
               normalize_obs: bool,
               standardization_stats_init_iter: int,
               standardize_obs: bool,
               raw_height: int,
               raw_width: int,
               resize_dim: Optional[tuple]=None):
    logger.info("Creating env...")
    
    default_transform = []
    
    if normalize_obs:
        default_transform.append(ToTensorImage(in_keys=["pixels"], out_keys=["pixels_transformed"]))
        default_transform.append(ToTensorImage(in_keys=["goal_pixels"], out_keys=["goal_pixels_transformed"]))
    
    if resize_dim is not None:
        default_transform.append(Resize(w=resize_dim[0], h=resize_dim[1], in_keys=["pixels_transformed"], out_keys=["pixels_transformed"]))
        default_transform.append(Resize(w=resize_dim[0], h=resize_dim[1], in_keys=["goal_pixels_transformed"], out_keys=["goal_pixels_transformed"]))
    
    if standardize_obs:
        assert standardization_stats_init_iter > 0, "standardization_stats_init_iter should be > 0 otherwise we can't compute the stats for standardization!"
        observation_norm = ObservationNorm(in_keys=["pixels_transformed"], out_keys=["pixels_transformed"], standard_normal=True)
        default_transform.append(observation_norm)
        
        logger.info("Computing observations standardization statistics...")
        observation_norm.init_stats(standardization_stats_init_iter)
        logger.info("Computed observations standardization statistics!")

        default_transform.append(ObservationNorm(loc=observation_norm.loc, scale=observation_norm.scale, in_keys=["goal_pixels_transformed"], out_keys=["goal_pixels_transformed"], standard_normal=True))
    
    default_transform = Compose(*default_transform)
    
    desired_goal_shape = None
    if env_name == "AntMaze_UMaze-v4":
        env, goal_strategy = create_ant_maze_env(device)
    elif env_name == "FrankaKitchen-v1":
        env, goal_strategy = create_franka_kitchen_env(device)
    elif env_name == "PointMaze_UMaze-v3":
        env, goal_strategy = create_point_maze_env(device)
    elif env_name == "AdroitHandRelocate-v1":
        env, goal_strategy = create_androit_hand_relocate_env(device)
    else:
        raise ValueError(f"Unknown environment name: '{env_name}'")
    
    env = GoalEnv(env=env, 
                  raw_obs_height=raw_height, 
                  raw_obs_width=raw_width,
                  env_goal_strategy=goal_strategy)
    
    env = TransformedEnv(env, default_transform)
    
    logger.info("Checking env specs...")
    check_env_specs(env)
    
    logger.info("Env created!")
    set_seed(env, seed)
    
    return env


def create_franka_kitchen_env(device: torch.device) -> tuple:
    env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave'], render_mode='rgb_array')
    env = GymWrapper(env, from_pixels=True, pixels_only=False, device=device)
    goal_strategy = FrankaKitchenEnvGoalStrategy(task_name="microwave")
    return env, goal_strategy


def create_ant_maze_env(device: torch.device) -> tuple:
    env = gym.make('AntMaze_UMazeDense-v4', render_mode='rgb_array')
    env = GymWrapper(env, from_pixels=True, pixels_only=False, device=device)
    goal_strategy = AntMazeEnvGoalStrategy()
    return env, goal_strategy


def create_point_maze_env(device: torch.device) -> tuple:
    env = gym.make('PointMaze_UMazeDense-v3', render_mode='rgb_array')
    env = GymWrapper(env, from_pixels=True, pixels_only=False, device=device)
    goal_strategy = PointMazeEnvGoalStrategy()
    return env, goal_strategy


def create_androit_hand_relocate_env(device: torch.device) -> tuple:
    env = gym.make('AdroitHandRelocate-v1', render_mode='rgb_array', camera_name="free")
    env.mujoco_renderer.default_cam_config['distance'] = 0.8
    env = GymWrapper(env, from_pixels=True, pixels_only=False, device=device)
    env.unwrapped.mujoco_renderer._set_cam_config()
    
    env = TransformedEnv(env)
    
    
    # Removes goal-informed obs info
    env.append_transform(RemoveDataFromObservation(index_to_remove_from_obs=torch.arange(start=30, end=39, step=1).tolist(),
                                                   original_obs_nb_dims=39))
    goal_strategy = AndroitHandRelocateEnvGoalStrategy()
    
    return env, goal_strategy


def set_seed(env, seed):
    env.set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
