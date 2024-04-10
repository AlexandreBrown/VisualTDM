from torchrl.envs import EnvBase
from typing import Optional
from torchrl.data import BoundedTensorSpec


class GoalEnv(EnvBase):
    def __init__(self, env, obs_height, obs_width):
        super().__init__(device=env.device, batch_size=env.batch_size)
        self.env = env
        self.observation_spec = self.env.observation_spec.clone()
        self.action_spec = self.env.action_spec.clone()
        self.done_spec = self.env.done_spec.clone()
        self.state_spec = self.env.state_spec.clone()
        self.reward_spec = self.env.reward_spec.clone()
        # Goal is an image
        self.observation_spec['goal_pixels'] = BoundedTensorSpec(
            low=0,
            high=255,
            shape=(obs_width, obs_height, 3),
            device=self.env.device,
        )
    
    def _step(self, tensordict):
        return self.env._step(tensordict)

    def _reset(self, tensordict):
        tensordict = self.env._reset(tensordict)
        self.goal_position = tensordict["desired_goal"]
        
        original_agent_position = [self.env.unwrapped.ant_env.data.qpos[0], self.env.unwrapped.ant_env.data.qpos[1]]
        
        # Changes agent position to the goal
        self.env.unwrapped.ant_env.data.qpos[0] = self.goal_position[0]
        self.env.unwrapped.ant_env.data.qpos[1] = self.goal_position[1]
        # Update env
        goal_tensordict = self.env.rand_step()
        # Take a snapeshot of the goal pixels observation
        goal_pixels = goal_tensordict['next']['pixels']
        tensordict["goal_pixels"] = goal_pixels
        
        # Put agent back to his original position
        self.env.unwrapped.ant_env.data.qpos[0] = original_agent_position[0]
        self.env.unwrapped.ant_env.data.qpos[1] = original_agent_position[1]
        # Update env
        tensordict = self.env.rand_step(tensordict)
        
        tensordict['next']['goal_pixels'] = goal_pixels
        
        return tensordict
    
    def _set_seed(self, seed: Optional[int]):
        self.env._set_seed(seed)
