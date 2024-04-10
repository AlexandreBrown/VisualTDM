import logging
import gymnasium as gym
import torch
import numpy as np
import random
from torchrl.envs.transforms import TransformedEnv
from torchrl.envs.transforms import Compose
from torchrl.envs.transforms import ToTensorImage
from torchrl.envs.transforms import Resize
from torchrl.envs.transforms import Transform
from torchrl.envs.transforms import ObservationNorm
from torchrl.envs import GymWrapper
from typing import Optional


logger = logging.getLogger(__name__)


def create_env(env_name: str,
               seed: int,
               device: torch.device,
               normalize_obs: bool,
               standardization_stats_init_iter: int,
               standardize_obs: bool,
               resize_dim: Optional[tuple]=None):
    logger.info("Creating env...")
    
    default_transform = []
    
    if normalize_obs:
        default_transform.append(ToTensorImage(in_keys=["pixels"], out_keys=["pixels_transformed"]))
    
    if resize_dim is not None:
        default_transform.append(Resize(w=resize_dim[0], h=resize_dim[1], in_keys=["pixels_transformed"], out_keys=["pixels_transformed"]))
    
    if standardize_obs:
        observation_norm = ObservationNorm(in_keys=["pixels_transformed"], out_keys=["pixels_transformed"], standard_normal=True)
        default_transform.append(observation_norm)
    
    default_transform = Compose(*default_transform)
    
    if env_name == "AntMaze_UMaze-v4":
        env = create_ant_maze_env(default_transform, device)
    elif env_name == "FrankaKitchen-v1":
        env = create_franka_kitchen_env(default_transform, device)
    else:
        raise ValueError(f"Unknown environment name: '{env_name}'")
    
    logger.info("Env created!")
    
    set_seed(env, seed)
    
    if standardize_obs and standardization_stats_init_iter > 0:
        logger.info("Computing observations standardization statistics...")
        observation_norm.init_stats(standardization_stats_init_iter)
        logger.info("Computed observations standardization statistics!")
    
    return env


def create_franka_kitchen_env(default_transform: Transform, device: torch.device):
    env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave'], render_mode='rgb_array')
    env = GymWrapper(env, from_pixels=True, pixels_only=False, device=device)
    
    env = TransformedEnv(env, default_transform)
    
    return env


def create_ant_maze_env(default_transform: Transform, device: torch.device):
    env = gym.make('AntMaze_UMaze-v4', render_mode='rgb_array')
    env = GymWrapper(env, from_pixels=True, pixels_only=False, device=device)
    
    env = TransformedEnv(env, default_transform)
    
    return env

def set_seed(env, seed):
    env.set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
