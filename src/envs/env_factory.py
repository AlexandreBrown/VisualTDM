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
from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger
from torchrl.envs.libs.gym import GymWrapper
from typing import Optional


def create_env(exp_name: str,
               seed: int,
               env_name: str,
               video_dir: str, 
               video_fps: int,
               device: torch.device,
               resize_dim: Optional[tuple]=None):
    
    logger = create_video_logger(exp_name, video_dir, video_fps)
    
    default_transform = [
        VideoRecorder(logger=logger, tag="step"),
        ToTensorImage(in_keys=["pixels"], out_keys=["pixels_transformed"]),
    ]
    if resize_dim is not None:
        default_transform.append(Resize(w=resize_dim[0], h=resize_dim[1], in_keys=["pixels_transformed"], out_keys=["pixels_transformed"]))
    
    default_transform.append(ObservationNorm(in_keys=["pixels_transformed"], out_keys=["pixels_transformed"], standard_normal=True))
    
    default_transform = Compose(*default_transform)
    
    if env_name == "AntMaze_UMaze-v4":
        env = create_ant_maze_env(default_transform, device)
    elif env_name == "FetchPickAndPlace-v2":
        env = create_pick_and_place_env(default_transform, device)
    elif env_name == "FrankaKitchen-v1":
        env = create_franka_kitchen_env(default_transform, device)
    else:
        raise ValueError(f"Unknown environment name: '{env_name}'")
    
    set_seed(env, seed)
    
    return env


def create_video_logger(exp_name: str, video_dir: str, video_fps: int):
    return CSVLogger(exp_name=exp_name, log_dir=video_dir, video_format="mp4", video_fps=video_fps)


def create_franka_kitchen_env(default_transform: Transform, device: torch.device):
    env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave'], render_mode='rgb_array')
    env = GymWrapper(env, from_pixels=True, pixels_only=True, device=device)
    
    env = TransformedEnv(env, default_transform)
    
    return env


def create_pick_and_place_env(default_transform: Transform, device: torch.device):
    env = gym.make('FetchPickAndPlace-v2', render_mode='rgb_array')
    env = GymWrapper(env, from_pixels=True, pixels_only=True, device=device)
    
    env = TransformedEnv(env, default_transform)
    
    return env


def create_ant_maze_env(default_transform: Transform, device: torch.device):
    env = gym.make('AntMaze_UMaze-v4', render_mode='rgb_array')
    env = GymWrapper(env, from_pixels=True, pixels_only=True, device=device)
    
    env = TransformedEnv(env, default_transform)
    
    return env


def set_seed(env, seed):
    env.set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
