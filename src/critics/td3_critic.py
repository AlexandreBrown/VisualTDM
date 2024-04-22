import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import copy
from tensordict import TensorDict
from actors.td3_actor import Td3Actor
from tensor_utils import get_tensor
from models.mlps.simple_mlp import SimpleMlp
from envs.dimensions import get_dim


class Td3Critic(nn.Module):
    def __init__(self,
                 device: torch.device,
                 model_type: str,
                 critic_in_keys: list,
                 goal_dim: int,
                 state_dim: int,
                 actions_dim: int,
                 action_space_low: int,
                 action_space_high: int,
                 hidden_layers_out_features: list,
                 use_batch_norm: bool,
                 hidden_activation_function_name: str,
                 output_activation_function_name: str,
                 learning_rate: float,
                 actor: Td3Actor,
                 actor_in_keys: list,
                 polyak_avg: float,
                 target_policy_action_noise_clip: float,
                 target_policy_action_noise_std: float,
                 gamma: float,
                 grad_norm_clipping: float):
        super().__init__()
        self.device = device
        self.model_type = model_type
        self.critic_in_keys = critic_in_keys
        self.state_dim = state_dim
        self.actions_dim = actions_dim
        self.goal_dim = goal_dim
        self.action_space_low = action_space_low.to(device)
        self.action_space_high = action_space_high.to(device)
        self.input_dim = sum([get_dim(self, key) for key in critic_in_keys])
        self.actor = actor
        self.actor_in_keys = actor_in_keys
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.polyak_avg = polyak_avg
        self.target_policy_action_noise_clip = target_policy_action_noise_clip
        self.target_policy_action_noise_std = target_policy_action_noise_std
        self.gamma = gamma
        self.grad_norm_clipping = grad_norm_clipping
        self.qf1 = SimpleMlp(input_dim=self.input_dim,
                                  hidden_layers_out_features=hidden_layers_out_features,
                                  use_batch_norm=use_batch_norm,
                                  hidden_activation_function_name=hidden_activation_function_name,
                                  output_activation_function_name=output_activation_function_name,
                                  out_dim=1).to(device)
        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=learning_rate)
        self.qf1_target = copy.deepcopy(self.qf1).to(device)
        self.qf1_loss_fn = F.mse_loss
        
        self.qf2 = SimpleMlp(input_dim=self.input_dim,
                                  hidden_layers_out_features=hidden_layers_out_features,
                                  use_batch_norm=use_batch_norm,
                                  hidden_activation_function_name=hidden_activation_function_name,
                                  output_activation_function_name=output_activation_function_name,
                                  out_dim=1).to(device)
        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=learning_rate)
        self.qf2_target = copy.deepcopy(self.qf2).to(device)
        self.qf2_loss_fn = F.mse_loss
    
    def update(self, train_data: TensorDict) -> dict:
        q_target = self.compute_target(train_data)
        
        x = get_tensor(train_data, self.critic_in_keys)
        
        qf1_values = self.qf1(x)
        qf1_loss = self.qf1_loss_fn(qf1_values, q_target)
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        nn.utils.clip_grad_value_(self.qf1.parameters(), self.grad_norm_clipping)
        self.qf1_optimizer.step()
        
        qf2_values = self.qf2(x)
        qf2_loss = self.qf2_loss_fn(qf2_values, q_target)
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        nn.utils.clip_grad_value_(self.qf2.parameters(), self.grad_norm_clipping)
        self.qf2_optimizer.step()
        
        return {
            'critic_loss_1':qf1_loss.item(),
            'critic_loss_2':qf2_loss.item()
        }
    
    def compute_target(self, train_data: TensorDict) -> torch.Tensor:
        actor_inputs = self.get_actor_inputs_for_next_action(train_data)
        next_action = self.actor_target(actor_inputs, train=True)
        smoothed_next_action = self.add_noise(next_action)
        
        target_qf_inputs = self.get_target_qf_inputs(train_data, smoothed_next_action)
        target_qf1_values = self.qf1_target(target_qf_inputs)
        target_qf2_values = self.qf2_target(target_qf_inputs)
        target_q_values = torch.min(target_qf1_values, target_qf2_values)
        
        target = train_data['next']['reward'] + self.gamma * target_q_values * (1 - train_data['next']['done'].type(torch.float))
        target = target.detach()
        
        return target
    
    def get_actor_inputs_for_next_action(self, train_data: TensorDict) -> torch.Tensor:
        inputs = []
        for key in self.actor_in_keys:
            if key == "state":
                inputs.append(train_data['next'][key])
            elif key == "desired_goal":
                inputs.append(train_data[key])
        
        return torch.cat(inputs, dim=1)
    
    def add_noise(self, next_action: torch.Tensor) -> torch.Tensor:
        noise = torch.normal(mean=torch.zeros_like(next_action), std=self.target_policy_action_noise_std)
        noise = torch.clamp(noise, -self.target_policy_action_noise_clip, self.target_policy_action_noise_clip)
        smoothed_next_action = next_action + noise
        smoothed_next_action = torch.clamp(smoothed_next_action, self.action_space_low, self.action_space_high)
        return smoothed_next_action
    
    def get_target_qf_inputs(self, train_data: TensorDict, smoothed_next_action: torch.Tensor) -> torch.Tensor:
        target_net_inputs = []
        
        for key in self.critic_in_keys:
            if key == "state":
                target_net_inputs.append(train_data['next'][key])
            elif key == "action":
                target_net_inputs.append(smoothed_next_action)
            elif key == "desired_goal":
                target_net_inputs.append(train_data[key])
        
        return torch.cat(target_net_inputs, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qf1(x)
    
    def update_target_network(self):
        for target_param, param in zip(self.qf1_target.parameters(), self.qf1.parameters()):
            target_param.data.copy_(self.polyak_avg * param.data + (1 - self.polyak_avg) * target_param.data)
        
        for target_param, param in zip(self.qf2_target.parameters(), self.qf2.parameters()):
            target_param.data.copy_(self.polyak_avg * param.data + (1 - self.polyak_avg) * target_param.data)
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.polyak_avg * param.data + (1 - self.polyak_avg) * target_param.data)
