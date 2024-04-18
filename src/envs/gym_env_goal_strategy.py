import numpy as np
import torch
import copy
from tensordict import TensorDict


class AntMazeEnvGoalStrategy:
    def get_goal_data(self, env, tensordict: TensorDict) -> tuple[TensorDict, torch.Tensor]:
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
        
    def get_goal_data(self, env, tensordict: TensorDict) -> tuple[TensorDict, torch.Tensor]:
        goal_position = tensordict['desired_goal'][self.task_name]
        
        original_position = copy.deepcopy(env.unwrapped.robot_env.data.joint(self.task_name).qpos)
        
        env.unwrapped.robot_env.data.joint(self.task_name).qpos = goal_position
        
        goal_tensordict = env.rand_step()
        goal_pixels = goal_tensordict['next']['pixels']
        
        env.unwrapped.robot_env.data.joint(self.task_name).qpos = original_position
        
        tensordict = env.rand_step(tensordict)
        
        return tensordict, goal_pixels

class PointMazeEnvGoalStrategy:
    def get_goal_data(self, env, tensordict: TensorDict) -> tuple[TensorDict, torch.Tensor]:
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

class AndroitHandRelocateEnvGoalStrategy:
    def get_goal_data(self, env, tensordict: TensorDict) -> tuple[TensorDict, torch.Tensor]:
        goal_x = np.random.uniform(low=-0.2, high=0.2, size=(1,))
        goal_y = np.random.uniform(low=-0.2, high=0.3, size=(1,))
        goal_z = np.random.uniform(low=0.07, high=0.35, size=(1,))
        goal_position = np.array([goal_x[0], goal_y[0], goal_z[0]])
        
        env_state = env.unwrapped.get_env_state()
        original_qpos = copy.deepcopy(env_state['qpos'])
        original_qvel = copy.deepcopy(env_state['qvel'])
        original_obj_pos = copy.deepcopy(env_state['obj_pos'])

        env.unwrapped.set_env_state({
            'qpos': original_qpos,
            'qvel': original_qvel,
            'obj_pos': goal_position,
            'target_pos': goal_position,
        })
        goal_pixels = torch.from_numpy(env.unwrapped.render().copy())
        
        env.unwrapped.set_env_state({
            'qpos': original_qpos,
            'qvel': original_qvel,
            'obj_pos': original_obj_pos,
            'target_pos': goal_position,
        })
        pixels = torch.from_numpy(env.unwrapped.render().copy())
        tensordict['pixels'] = pixels
        
        self.desired_goal = torch.from_numpy(goal_position)
        
        return tensordict, goal_pixels
