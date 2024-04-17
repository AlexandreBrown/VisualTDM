import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from tensordict import TensorDict
from actors.tdm_actor import TdmActor
from models.resnets.mini_resnets import MiniResNet3
from models.mlps.simple_mlp import SimpleMlp
from rewards.distance import compute_distance
from tensor_utils import get_tensor
from envs.dimensions import get_dim

class TdmCritic(nn.Module):
    def __init__(self,
                 model_type: str,
                 obs_dim: int,
                 actions_dim: int,
                 goal_dim: int,
                 goal_latent_dim: int,
                 hidden_layers_out_features: list,
                 hidden_activation_function_name: str,
                 output_activation_function_name: str,
                 device: torch.device,
                 norm_type: str,
                 actor: TdmActor,
                 learning_rate: float,
                 polyak_avg: float,
                 target_policy_action_clip: float,
                 state_dim: int,
                 actor_in_keys: list,
                 critic_in_keys: list):
        super().__init__()
        self.actor_in_keys = actor_in_keys
        self.critic_in_keys = critic_in_keys
        self.goal_dim = goal_dim
        self.goal_latent_dim = goal_latent_dim
        self.state_dim = state_dim
        self.actions_dim = actions_dim
        self.tau_dim = 1
        self.critic_input_dim = sum([get_dim(self, key) for key in critic_in_keys])
        if model_type == "mini_resnet_3":
            last_out_channels = 512
            assert "pixels_latent" not in self.critic_in_keys, "MiniResNet3 is expected to receive the observation image, not its latent representation!"
            assert "pixels_transformed" in self.critic_in_keys, "MinResNet3 needs the pixels_transformed in_key, it works on the obs image directly!"
            fc1_in_features = last_out_channels + self.critic_input_dim
            self.state_net = MiniResNet3(in_channels=obs_dim,
                                         fc1_in_features=fc1_in_features,
                                         fc1_out_features=hidden_layers_out_features[0],
                                         out_dim=goal_latent_dim)
        elif model_type == "mlp_pretrained_encoder":
            self.state_net = SimpleMlp(input_dim=self.critic_input_dim,
                                 hidden_layers_out_features=hidden_layers_out_features,
                                 hidden_activation_function_name=hidden_activation_function_name,
                                 output_activation_function_name=output_activation_function_name,
                                 out_dim=goal_latent_dim)
        else:
            raise ValueError(f"Unknown model type '{model_type}'!")
        self.state_net = self.state_net.to(device)
        self.state_net_target = copy.deepcopy(self.state_net).to(device)
        self.norm_type = norm_type
        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.learning_rate = learning_rate
        self.polyak_avg = polyak_avg
        self.optimizer = optim.Adam(self.state_net.parameters(), lr=learning_rate)
        self.device = device
        self.target_policy_action_clip = target_policy_action_clip
        goal_key_index = critic_in_keys.index('goal_latent')
        self.goal_latent_feature_index_start = sum([get_dim(self, key) for key in critic_in_keys[:goal_key_index]])
        self.goal_latent_feature_index_end = self.goal_latent_feature_index_start + goal_latent_dim
    
    def update(self, train_data: TensorDict) -> dict:
        y = self.compute_target(train_data)
        
        x = get_tensor(train_data, self.critic_in_keys)
        y_hat = self(x)

        loss = F.mse_loss(y_hat, y, reduction='sum') / train_data.batch_size[0]
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'critic_loss':loss.item()
        }
    
    def compute_target(self, train_data: TensorDict) -> torch.Tensor:
        
        target_train_data = copy.deepcopy(train_data)
        target_train_data['planning_horizon'][target_train_data['planning_horizon'] > 0] -= 1
        
        actor_inputs_next = get_tensor(target_train_data['next'], self.actor_in_keys[:2])
        actor_inputs_current = get_tensor(target_train_data, self.actor_in_keys[2:])
        actor_inputs = torch.cat([actor_inputs_next, actor_inputs_current], dim=1)
        next_actions = self.actor_target(actor_inputs).clip(min=-self.target_policy_action_clip, max=self.target_policy_action_clip)

        target_train_data['action'] = next_actions
        target_inputs = get_tensor(target_train_data, self.critic_in_keys)
        target_q_values = self.state_net_target(target_inputs)
        
        rewards = target_train_data['next']['reward']
        done = target_train_data['next']['done'].type(torch.uint8)
        target = rewards + (1. - done) * target_q_values
        target = target.detach()
        
        return target
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs_latent = self.state_net(x)
        
        goal_latent = x[:, self.goal_latent_feature_index_start:self.goal_latent_feature_index_end]
        
        distance = compute_distance(norm_type=self.norm_type,
                                    obs_latent=obs_latent,
                                    goal_latent=goal_latent)
        
        return -distance
    
    def update_target_network(self):
        for target_param, param in zip(self.state_net_target.parameters(), self.state_net.parameters()):
            target_param.data.copy_(self.polyak_avg * param.data + (1 - self.polyak_avg) * target_param.data)
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.polyak_avg * param.data + (1 - self.polyak_avg) * target_param.data)
