from torchrl.collectors.collectors import SyncDataCollector
from torchrl.collectors.collectors import RandomPolicy
from tqdm import tqdm
from envs.env_factory import create_env


def explore_env(exp_name: str,
                seed: int,
                env_name: str,
                video_dir: str, 
                video_fps: int):

    env = create_env(exp_name=exp_name,
                     seed=seed,
                     env_name=env_name,
                     video_dir=video_dir,
                     video_fps=video_fps)
    
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
