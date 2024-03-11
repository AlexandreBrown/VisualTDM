import gymnasium as gym
import torch
import numpy as np
import random
from torchrl.envs.transforms import ToTensorImage, TransformedEnv
from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger
from torchrl.envs.libs.gym import GymWrapper


def create_env(exp_name: str,
               seed: int,
               env_name: str,
               video_dir: str, 
               video_fps: int):
    if env_name == "FrankaKitchen-v1":
        return create_franka_kitchen_env(exp_name,
                                         seed,
                                         video_dir,
                                         video_fps)
    elif env_name == "FetchPickAndPlace-v2":
        return create_pick_and_place_env(exp_name,
                                         seed,
                                         video_dir,
                                         video_fps)
    elif env_name == "AntMaze_UMaze-v4":
        return create_ant_maze_env(exp_name,
                                   seed,
                                   video_dir,
                                   video_fps)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")


def create_franka_kitchen_env(exp_name: str, seed: int, video_dir: str, video_fps: int):
    
    env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave'], render_mode='rgb_array')
    env = GymWrapper(env, from_pixels=True,pixels_only=True)
    
    logger = CSVLogger(exp_name=exp_name, log_dir=video_dir, video_format="mp4", video_fps=video_fps)
    
    env = TransformedEnv(env)
    env.append_transform(VideoRecorder(logger=logger, tag="step"))
    env.append_transform(ToTensorImage())
    
    set_seed(env, seed)
    
    return env


def create_pick_and_place_env(exp_name: str, seed: int, video_dir: str, video_fps: int):
    
    env = gym.make('FetchPickAndPlace-v2', render_mode='rgb_array')
    env = GymWrapper(env, from_pixels=True,pixels_only=True)
    
    logger = CSVLogger(exp_name=exp_name, log_dir=video_dir, video_format="mp4", video_fps=video_fps)
    
    env = TransformedEnv(env)
    env.append_transform(VideoRecorder(logger=logger, tag="step"))
    env.append_transform(ToTensorImage())
    
    set_seed(env, seed)
    
    return env

def create_ant_maze_env(exp_name: str, seed: int, video_dir: str, video_fps: int):
    
    env = gym.make('AntMaze_UMaze-v4', render_mode='rgb_array')
    env = GymWrapper(env, from_pixels=True,pixels_only=True)
    
    logger = CSVLogger(exp_name=exp_name, log_dir=video_dir, video_format="mp4", video_fps=video_fps)
    
    env = TransformedEnv(env)
    env.append_transform(VideoRecorder(logger=logger, tag="step"))
    env.append_transform(ToTensorImage())
    
    set_seed(env, seed)
    
    return env

def set_seed(env, seed):
    env.set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
