
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.collectors.collectors import RandomPolicy
from tqdm import tqdm
from functools import partial
from envs.env_factory import create_franka_kitchen_env


if __name__ == "__main__":
    seed = 42
    video_fps = 30
    
    env_factory = partial(create_franka_kitchen_env, video_fps=video_fps, seed=seed)
    
    env = env_factory()
    
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
        continue
    
    print("Done exploring env!")
    
    print("Closing env...")
    env.transform.dump()
    env.close()
