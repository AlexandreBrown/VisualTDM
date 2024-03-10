import gymnasium as gym

from torchrl.envs.transforms import ToTensorImage, TransformedEnv
from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger
from torchrl.envs.libs.gym import GymWrapper


def create_env(env_name: str, video_fps: int, seed: int):
    if env_name == "FrankaKitchen":
        return create_franka_kitchen_env(video_fps, seed)
    elif env_name == "PickAndPlace":
        return create_pick_and_place_env(video_fps, seed)
    elif env_name == "AntMaze_UMaze":
        return create_ant_maze_env(video_fps, seed)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")


def create_franka_kitchen_env(video_fps: int, seed: int):
    
    env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave'], render_mode='rgb_array')
    env = GymWrapper(env, from_pixels=True,pixels_only=True)
    
    logger = CSVLogger(exp_name="FrankaKitchen", log_dir="kitchen_videos", video_format="mp4", video_fps=video_fps)
    
    env = TransformedEnv(env)
    env.append_transform(VideoRecorder(logger=logger, tag="iteration"))
    env.append_transform(ToTensorImage())
    
    env.set_seed(seed)
    
    return env


def create_pick_and_place_env(video_fps: int, seed: int):
    
    env = gym.make('FetchPickAndPlace-v2', render_mode='rgb_array')
    env = GymWrapper(env, from_pixels=True,pixels_only=True)
    
    logger = CSVLogger(exp_name="FetchPickAndPlace", log_dir="pick_and_place_videos", video_format="mp4", video_fps=video_fps)
    
    env = TransformedEnv(env)
    env.append_transform(VideoRecorder(logger=logger, tag="iteration"))
    env.append_transform(ToTensorImage())
    
    env.set_seed(seed)
    
    return env

def create_ant_maze_env(video_fps: int, seed: int):
    
    env = gym.make('AntMaze_UMaze-v4', render_mode='rgb_array')
    env = GymWrapper(env, from_pixels=True,pixels_only=True)
    
    logger = CSVLogger(exp_name="AntMaze_UMaze", log_dir="logs/ant_maze_umaze_videos", video_format="mp4", video_fps=video_fps)
    
    env = TransformedEnv(env)
    env.append_transform(VideoRecorder(logger=logger, tag="iteration"))
    env.append_transform(ToTensorImage())
    
    env.set_seed(seed)
    
    return env
