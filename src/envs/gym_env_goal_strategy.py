import torch
import copy
from tensordict import TensorDict


class AntMazeEnvGoalStrategy:
    def get_goal_pixels(self, env, tensordict: TensorDict) -> tuple[TensorDict, torch.Tensor]:
        goal_position = tensordict["desired_goal"]
        
        original_agent_position = copy.deepcopy(env.unwrapped.ant_env.data.qpos)
        
        env.unwrapped.ant_env.data.qpos[0] = goal_position[0]
        env.unwrapped.ant_env.data.qpos[1] = goal_position[1]
        
        goal_tensordict = env.rand_step()
        goal_pixels = goal_tensordict['next']['pixels']
        
        env.unwrapped.ant_env.data.qpos[0] = original_agent_position[0]
        env.unwrapped.ant_env.data.qpos[1] = original_agent_position[1]
        
        tensordict = env.rand_step(tensordict)
        
        return tensordict, goal_pixels


class FrankaKitchenEnvGoalStrategy:
    def __init__(self, task_name: str):
        self.task_name = task_name
        
    def get_goal_pixels(self, env, tensordict: TensorDict) -> tuple[TensorDict, torch.Tensor]:
        goal_position = tensordict['desired_goal'][self.task_name]
        
        original_position = copy.deepcopy(env.unwrapped.robot_env.data.joint(self.task_name).qpos)
        
        env.unwrapped.robot_env.data.joint(self.task_name).qpos = goal_position
        
        goal_tensordict = env.rand_step()
        goal_pixels = goal_tensordict['next']['pixels']
        
        env.unwrapped.robot_env.data.joint(self.task_name).qpos = original_position
        
        tensordict = env.rand_step(tensordict)
        
        return tensordict, goal_pixels

class PointMazeEnvGoalStrategy:
    def get_goal_pixels(self, env, tensordict: TensorDict) -> tuple[TensorDict, torch.Tensor]:
        goal_position = tensordict["desired_goal"]
        
        original_agent_position = copy.deepcopy(env.unwrapped.point_env.data.qpos)
        
        env.unwrapped.point_env.data.qpos[0] = goal_position[0]
        env.unwrapped.point_env.data.qpos[1] = goal_position[1]
        
        goal_tensordict = env.rand_step()
        goal_pixels = goal_tensordict['next']['pixels']
        
        env.unwrapped.point_env.data.qpos[0] = original_agent_position[0]
        env.unwrapped.point_env.data.qpos[1] = original_agent_position[1]
        
        tensordict = env.rand_step(tensordict)
        
        return tensordict, goal_pixels