import numpy as np
import torch
import copy
from tensordict import TensorDict
from torchrl.envs import step_mdp
from torchrl.envs import EnvBase


class AntMazeEnvGoalStrategy:
    def get_goal_data(self, env: EnvBase, tensordict: TensorDict) -> tuple[TensorDict, torch.Tensor]:
        goal_position = tensordict["desired_goal"]
        
        original_agent_position = copy.deepcopy(env.unwrapped.ant_env.data.qpos)
        
        env.unwrapped.ant_env.data.qpos[0] = goal_position[0]
        env.unwrapped.ant_env.data.qpos[1] = goal_position[1]
        
        tensordict_with_action = TensorDict(source={'action':torch.zeros(size=env.action_spec.shape)}, batch_size=[])
        goal_tensordict = step_mdp(env.step(tensordict_with_action))
        goal_pixels = goal_tensordict['pixels']
        
        env.unwrapped.ant_env.data.qpos[0] = original_agent_position[0]
        env.unwrapped.ant_env.data.qpos[1] = original_agent_position[1]
        
        tensordict['action'] = torch.zeros(size=env.action_spec.shape)
        tensordict = step_mdp(env.step(tensordict))
        
        return tensordict, goal_pixels


class PointMazeEnvGoalStrategy:
    def get_goal_data(self, env, tensordict: TensorDict) -> tuple[TensorDict, torch.Tensor]:
        goal_position = tensordict["desired_goal"]
        
        original_agent_position = copy.deepcopy(env.unwrapped.point_env.data.qpos)
        
        env.unwrapped.point_env.data.qpos[0] = goal_position[0]
        env.unwrapped.point_env.data.qpos[1] = goal_position[1]
        
        tensordict_with_action = TensorDict(source={'action':torch.zeros(size=env.action_spec.shape)}, batch_size=[])
        goal_tensordict = step_mdp(env.step(tensordict_with_action))
        goal_pixels = goal_tensordict['pixels']
        
        env.unwrapped.point_env.data.qpos[0] = original_agent_position[0]
        env.unwrapped.point_env.data.qpos[1] = original_agent_position[1]
        
        tensordict['action'] = torch.zeros(size=env.action_spec.shape)
        tensordict = step_mdp(env.step(tensordict))
        
        return tensordict, goal_pixels


class FrankaKitchenEnvGoalStrategy:
    def __init__(self, task_name: str):
        self.task_name = task_name
        
    def get_goal_data(self, env, tensordict: TensorDict) -> tuple[TensorDict, torch.Tensor]:
        goal_position = tensordict['desired_goal'][self.task_name]
        
        original_position = copy.deepcopy(env.unwrapped.robot_env.data.joint(self.task_name).qpos)
        
        env.unwrapped.robot_env.data.joint(self.task_name).qpos = goal_position
        
        tensordict_with_action = TensorDict(source={'action':torch.zeros(size=env.action_spec.shape)}, batch_size=[])
        goal_tensordict = step_mdp(env.step(tensordict_with_action))
        goal_pixels = goal_tensordict['pixels']
        
        env.unwrapped.robot_env.data.joint(self.task_name).qpos = original_position
        
        tensordict['action'] = torch.zeros(size=env.action_spec.shape)
        tensordict = step_mdp(env.step(tensordict))
        
        return tensordict, goal_pixels


class AndroitHandRelocateEnvGoalStrategy:
    def __init__(self, target_x_min_max: list, target_y_min_max: list, target_z_min_max: list):
        self.target_x_min_max = target_x_min_max
        self.target_y_min_max = target_y_min_max
        self.target_z_min_max = target_z_min_max
    
    def get_goal_data(self, env, tensordict: TensorDict) -> tuple[TensorDict, torch.Tensor]:
        goal_x = np.random.uniform(low=self.target_x_min_max[0], high=self.target_x_min_max[1], size=(1,))
        goal_y = np.random.uniform(low=self.target_y_min_max[0], high=self.target_y_min_max[1], size=(1,))
        goal_z = np.random.uniform(low=self.target_z_min_max[0], high=self.target_z_min_max[1], size=(1,))
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


class FetchReachEnvGoalStrategy:
    def __init__(self, 
                 target_x_min_max: list,
                 target_y_min_max: list,
                 target_z_min_max: list):
        self.target_x_min_max = target_x_min_max
        self.target_y_min_max = target_y_min_max
        self.target_z_min_max = target_z_min_max
        self.tolerance = 0.05
        self.momentum = 40
    
    def get_goal_data(self, env, tensordict: TensorDict) -> tuple[TensorDict, torch.Tensor]:
        goal_pos = self.sample_goal(env)
        original_grip_pos = copy.deepcopy(self.get_gripper_pos(env))
        
        env.unwrapped.goal = goal_pos
        
        tensordict = self.move_grip_to_destination(env, tensordict, destination_pos=goal_pos)
        
        goal_pixels = tensordict['pixels']
        
        tensordict = self.move_grip_to_destination(env, tensordict, destination_pos=original_grip_pos)

        return tensordict, goal_pixels

    def sample_goal(self, env) -> np.ndarray:
        gripper_pos = self.get_gripper_pos(env)
        x = np.random.uniform(low=self.target_x_min_max[0], high=self.target_x_min_max[1])
        y = np.random.uniform(low=self.target_y_min_max[0], high=self.target_y_min_max[1])
        z = np.random.uniform(low=self.target_z_min_max[0], high=self.target_z_min_max[1])
        return np.array([gripper_pos[0]+x, gripper_pos[1]+y, gripper_pos[2]+z])
    
    def get_gripper_pos(self, env) -> np.ndarray:
        return env.unwrapped.data.site('robot0:grip').xpos
    
    def move_grip_to_destination(self, env, tensordict: TensorDict, destination_pos: np.ndarray) -> TensorDict:
        gripper_pos = self.get_gripper_pos(env)
        
        momemtum_before = self.momentum
        
        while np.linalg.norm(destination_pos - gripper_pos, ord=2) > self.tolerance:
            action = torch.zeros((4,))
            if self.momentum > 0.01:
                action[:3] = torch.tanh(torch.from_numpy(destination_pos - gripper_pos)) * self.momentum
                self.momentum -= self.momentum * 0.05
            else:
                action[:3] = torch.from_numpy(destination_pos - gripper_pos)
            tensordict['action'] = action
            tensordict = step_mdp(env.step(tensordict))
            gripper_pos = self.get_gripper_pos(env)
        
        self.momentum = momemtum_before
        
        return tensordict


class InvertedPendulumEnvGoalStrategy:
    def get_goal_data(self, env, tensordict: TensorDict) -> tuple[TensorDict, torch.Tensor]:
        goal_pixels = tensordict['pixels']
        return tensordict, goal_pixels
