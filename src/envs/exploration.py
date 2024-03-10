
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.collectors.collectors import RandomPolicy
from tqdm import tqdm
from envs.env_factory import create_env


def explore_env(env_name: str, video_fps: int=30, seed: int=42):
    
    env = create_env(env_name=env_name, video_fps=video_fps, seed=seed)
    
    policy = RandomPolicy(action_spec=env.action_spec)
    
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=policy,
        total_frames=400,
        max_frames_per_traj=100,
        frames_per_batch=200,
        reset_at_each_iter=False,
        device="cpu",
        storing_device="cpu",
    )

    print("Exploring env...")
    
    for data in tqdm(collector):
        img = data['pixels'][0]
        continue
    
    print("Done exploring env!")
    
    print("Closing env...")
    env.transform.dump()
    env.close()
