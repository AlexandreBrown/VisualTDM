import gymnasium as gym

from torchrl.envs.transforms import ToTensorImage, TransformedEnv
from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger
from torchrl.envs.libs.gym import GymWrapper


def create_franka_kitchen_env(video_fps: int, seed: int):
    
    env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave'], render_mode='rgb_array')
    env = GymWrapper(env, from_pixels=True,pixels_only=True)
    
    logger = CSVLogger(exp_name="FrankaKitchen", log_dir="kitchen_videos", video_format="mp4", video_fps=video_fps)
    
    env = TransformedEnv(env)
    env.append_transform(VideoRecorder(logger=logger, tag="iteration"))
    env.append_transform(ToTensorImage())
    
    env.set_seed(seed)
    
    return env