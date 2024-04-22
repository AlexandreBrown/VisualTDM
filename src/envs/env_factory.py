import logging
import gymnasium as gym
import torch
import numpy as np
import random
from omegaconf import DictConfig
from torchrl.envs.transforms import TransformedEnv
from torchrl.envs.transforms import Compose
from torchrl.envs import GymWrapper
from torchrl.envs.utils import check_env_specs
from torchrl.envs import EnvBase


logger = logging.getLogger(__name__)


def create_env(cfg: DictConfig):
    logger.info("Creating env...")
    
    env_name = cfg['env']['name']
    seed = cfg['experiment']['seed']
    device = torch.device(cfg['env']['device'])
    
    default_transform = []
    default_transform = Compose(*default_transform)
    
    if env_name == "AntMaze_UMaze-v4":
        env = create_ant_maze_env(device, cfg)
    elif env_name == "PointMaze_UMaze-v3":
        env = create_point_maze_env(device, cfg)
    elif env_name == "FrankaKitchen-v1":
        env = create_franka_kitchen_env(device, cfg)
    elif env_name == "AdroitHandRelocate-v1":
        env = create_androit_hand_relocate_env(device, cfg)
    elif env_name == "FetchReach-v2":
        env = create_fetch_reach_env(device, cfg)
    elif env_name == "InvertedPendulum-v4":
        env = create_inverted_pendulum_env(device, cfg)
    else:
        raise ValueError(f"Unknown environment name: '{env_name}'")
    
    env = TransformedEnv(env, default_transform)
    
    logger.info("Checking env specs...")
    check_env_specs(env)
    
    logger.info("Env created!")
    set_seed(env, seed)
    
    return env


def create_ant_maze_env(device: torch.device, cfg: DictConfig) -> EnvBase:
    env = gym.make('AntMaze_UMazeDense-v4', render_mode='rgb_array', max_episode_steps=cfg['env']['max_frames_per_traj'])
    env = GymWrapper(env, from_pixels=True, pixels_only=False, device=device)
    return env


def create_point_maze_env(device: torch.device, cfg: DictConfig) -> EnvBase:
    env = gym.make('PointMaze_UMazeDense-v3', render_mode='rgb_array', max_episode_steps=cfg['env']['max_frames_per_traj'])
    env = GymWrapper(env, from_pixels=True, pixels_only=False, device=device)
    return env


def create_franka_kitchen_env(device: torch.device, cfg: DictConfig) -> EnvBase:
    env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave'], render_mode='rgb_array', max_episode_steps=cfg['env']['max_frames_per_traj'])
    env = GymWrapper(env, from_pixels=True, pixels_only=False, device=device)
    return env


def create_androit_hand_relocate_env(device: torch.device, cfg: DictConfig) -> EnvBase:
    env = gym.make('AdroitHandRelocate-v1', render_mode='rgb_array', max_episode_steps=cfg['env']['max_frames_per_traj'], camera_name="free")
    env = GymWrapper(env, from_pixels=True, pixels_only=False, device=device)
    return env


def create_fetch_reach_env(device: torch.device, cfg: DictConfig) -> EnvBase:
    env = gym.make('FetchReachDense-v2', render_mode='rgb_array', max_episode_steps=cfg['env']['max_frames_per_traj'])
    env = GymWrapper(env, from_pixels=True, pixels_only=False, device=device)

    return env


def create_inverted_pendulum_env(device: torch.device, cfg: DictConfig) -> EnvBase:
    env = gym.make('InvertedPendulum-v4', render_mode='rgb_array', max_episode_steps=cfg['env']['max_frames_per_traj'])
    env = GymWrapper(env, from_pixels=True, pixels_only=False, device=device)
    return env


def set_seed(env, seed):
    env.set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
