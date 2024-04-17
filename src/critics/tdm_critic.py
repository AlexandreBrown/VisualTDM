import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tensordict import TensorDict
from actors.tdm_actor import TdmTd3Actor
from models.resnets.mini_resnets import MiniResNet3
from models.mlps.simple_mlp import SimpleMlp
from rewards.distance import compute_distance
from tensor_utils import get_tensor
from envs.dimensions import get_dim

class TdmTd3Critic(nn.Module):
    def __init__(self,
                 model_type: str,
                 obs_dim: int,
                 actions_dim: int,
                 goal_dim: int,
                 goal_latent_dim: int,
                 hidden_layers_out_features: list,
                 use_batch_norm: bool,
                 hidden_activation_function_name: str,
                 output_activation_function_name: str,
                 device: torch.device,
                 norm_type: str,
                 actor: TdmTd3Actor,
                 learning_rate: float,
                 polyak_avg: float,
                 target_policy_action_noise_clip: float,
                 target_policy_action_noise_std: float,
                 state_dim: int,
                 actor_in_keys: list,
                 critic_in_keys: list,
                 action_space_low: torch.Tensor,
                 action_space_high: torch.Tensor):
        super().__init__()
        self.actor_in_keys = actor_in_keys
        self.critic_in_keys = critic_in_keys
        self.goal_dim = goal_dim
        self.goal_latent_dim = goal_latent_dim
        self.state_dim = state_dim
        self.actions_dim = actions_dim
        self.action_space_low = action_space_low.to(device)
        self.action_space_high = action_space_high.to(device)
        self.tau_dim = 1
        self.critic_input_dim = sum([get_dim(self, key) for key in critic_in_keys])
        self.state_net_1 = self.create_state_net(model_type, obs_dim, hidden_layers_out_features, goal_latent_dim, use_batch_norm, hidden_activation_function_name, output_activation_function_name).to(device)
        self.state_net_1_optimizer = optim.Adam(self.state_net_1.parameters(), lr=learning_rate)
        self.state_net_target_1 = copy.deepcopy(self.state_net_1).to(device)
        self.state_net_2 = self.create_state_net(model_type, obs_dim, hidden_layers_out_features, goal_latent_dim, use_batch_norm, hidden_activation_function_name, output_activation_function_name).to(device)
        self.state_net_2_optimizer = optim.Adam(self.state_net_2.parameters(), lr=learning_rate)
        self.state_net_target_2 = copy.deepcopy(self.state_net_2).to(device)
        self.norm_type = norm_type
        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.learning_rate = learning_rate
        self.polyak_avg = polyak_avg
        self.device = device
        self.target_policy_action_noise_clip = target_policy_action_noise_clip
        self.target_policy_action_noise_std = target_policy_action_noise_std
        goal_key_index = critic_in_keys.index('goal_latent')
        self.goal_latent_feature_index_start = sum([get_dim(self, key) for key in critic_in_keys[:goal_key_index]])
        self.goal_latent_feature_index_end = self.goal_latent_feature_index_start + goal_latent_dim
    
    def create_state_net(self, model_type, obs_dim, hidden_layers_out_features, goal_latent_dim, use_batch_norm, hidden_activation_function_name, output_activation_function_name):
        if model_type == "mini_resnet_3":
            last_out_channels = 512
            assert "pixels_latent" not in self.critic_in_keys, "MiniResNet3 is expected to receive the observation image, not its latent representation!"
            assert "pixels_transformed" in self.critic_in_keys, "MinResNet3 needs the pixels_transformed in_key, it works on the obs image directly!"
            fc1_in_features = last_out_channels + self.critic_input_dim
            state_net = MiniResNet3(in_channels=obs_dim,
                                         fc1_in_features=fc1_in_features,
                                         fc1_out_features=hidden_layers_out_features[0],
                                         out_dim=goal_latent_dim)
        elif model_type == "mlp_pretrained_encoder":
            state_net = SimpleMlp(input_dim=self.critic_input_dim,
                                 hidden_layers_out_features=hidden_layers_out_features,
                                 use_batch_norm=use_batch_norm,
                                 hidden_activation_function_name=hidden_activation_function_name,
                                 output_activation_function_name=output_activation_function_name,
                                 out_dim=goal_latent_dim)
        else:
            raise ValueError(f"Unknown model type '{model_type}'!")
        
        return state_net
    
    def update(self, train_data: TensorDict) -> dict:
        q_target = self.compute_target(train_data)
        
        x = get_tensor(train_data, self.critic_in_keys)
        
        q1 = self.compute_q_values(x, state_net=self.state_net_1)
        q1_loss = ((q1 - q_target) ** 2).sum(dim=1).mean()
        self.state_net_1_optimizer.zero_grad()
        q1_loss.backward()
        self.state_net_1_optimizer.step()
        
        q2 = self.compute_q_values(x, state_net=self.state_net_2)
        q2_loss = ((q2 - q_target) ** 2).sum(dim=1).mean()
        self.state_net_2_optimizer.zero_grad()
        q2_loss.backward()
        self.state_net_2_optimizer.step()
        
        return {
            'critic_loss_1':q1_loss.item(),
            'critic_loss_2':q2_loss.item()
        }
    
    def compute_target(self, train_data: TensorDict) -> torch.Tensor:
        
        target_train_data = train_data.clone(recurse=True)
        target_train_data['planning_horizon'][target_train_data['planning_horizon'] > 0] -= 1
        
        actor_inputs_next = get_tensor(target_train_data['next'], self.actor_in_keys[:2])
        actor_inputs_current = get_tensor(target_train_data, self.actor_in_keys[2:])
        actor_inputs = torch.cat([actor_inputs_next, actor_inputs_current], dim=1)
        
        next_actions = self.actor_target(actor_inputs)
        noise = torch.normal(mean=torch.zeros_like(next_actions), std=self.target_policy_action_noise_std)
        noise = torch.clamp(noise, -self.target_policy_action_noise_clip, self.target_policy_action_noise_clip)
        next_actions = next_actions + noise
        next_actions = torch.clamp(next_actions, self.action_space_low, self.action_space_high)
        
        target_train_data['action'] = next_actions
        target_inputs = get_tensor(target_train_data, self.critic_in_keys)
        target_q_values_1 = self.compute_q_values(target_inputs, self.state_net_target_1)
        target_q_values_2 = self.compute_q_values(target_inputs, self.state_net_target_2)
        target_q_values = torch.min(target_q_values_1, target_q_values_2)
        
        rewards = target_train_data['next']['reward']
        done = target_train_data['next']['done'].type(torch.uint8)
        target = rewards + (1. - done) * target_q_values
        target = target.detach()
        
        return target
    
    def compute_q_values(self, x: torch.Tensor, state_net: nn.Module) -> torch.Tensor:
        obs_latent = state_net(x)
        
        goal_latent = x[:, self.goal_latent_feature_index_start:self.goal_latent_feature_index_end]
        
        distance = compute_distance(norm_type=self.norm_type,
                                    obs_latent=obs_latent,
                                    goal_latent=goal_latent)
        
        return -distance
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compute_q_values(x, self.state_net_1)
    
    def update_target_network(self):
        for target_param, param in zip(self.state_net_target_1.parameters(), self.state_net_1.parameters()):
            target_param.data.copy_(self.polyak_avg * param.data + (1 - self.polyak_avg) * target_param.data)
        
        for target_param, param in zip(self.state_net_target_2.parameters(), self.state_net_2.parameters()):
            target_param.data.copy_(self.polyak_avg * param.data + (1 - self.polyak_avg) * target_param.data)
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.polyak_avg * param.data + (1 - self.polyak_avg) * target_param.data)
