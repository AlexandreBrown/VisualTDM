import torch
from torchrl.envs import EnvBase
from typing import Optional
from torchrl.data import BoundedTensorSpec


class GoalEnv(EnvBase):
    def __init__(self, 
                 env,
                 raw_obs_height: int, 
                 raw_obs_width: int,
                 env_goal_strategy):
        super().__init__(device=env.device, batch_size=env.batch_size)
        self.env = env
        self.observation_spec = env.observation_spec.clone()
        self.action_spec = env.action_spec.clone()
        self.done_spec = env.done_spec.clone()
        self.state_spec = env.state_spec.clone()
        self.reward_spec = env.reward_spec.clone()
        self.observation_spec['goal_pixels'] = BoundedTensorSpec(
            low=0,
            high=255,
            shape=(raw_obs_width, raw_obs_height, 3),
            device=env.device,
            dtype=torch.uint8
        )
        self.full_state_spec['goal_pixels'] = self.observation_spec['goal_pixels'].clone()
        self.env_goal_strategy = env_goal_strategy
    
    def _step(self, tensordict):
        goal_pixels = tensordict.get("goal_pixels")
        tensordict = self.env._step(tensordict)
        tensordict.set("goal_pixels", goal_pixels)
        return tensordict
        
    def _reset(self, tensordict):
        tensordict = self.env._reset(tensordict)
        tensordict, goal_pixels = self.env_goal_strategy.get_goal_pixels(self.env, tensordict)
        self.goal_pixels = goal_pixels
        tensordict["goal_pixels"] = goal_pixels
        tensordict['next']['goal_pixels'] = goal_pixels      
        return tensordict
        
    
    def _set_seed(self, seed: Optional[int]):
        self.env._set_seed(seed)
