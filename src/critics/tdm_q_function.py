import torch
import torch.nn as nn
from models.resnets.mini_resnets import MiniResNet3
from models.mlps.simple_mlp import SimpleMlp
from envs.dimensions import get_dim
from rewards.distance import compute_distance

class TdmQFunction(nn.Module):
    def __init__(self,
                 device: torch.device,
                 norm_type: str,
                 model_type: str,
                 in_keys: list,
                 goal_latent_dim: int,
                 state_dim: int,
                 actions_dim: int,
                 obs_dim: int,
                 action_space_low: float,
                 action_space_high: float,
                 hidden_layers_out_features: list,
                 use_batch_norm: bool,
                 hidden_activation_function_name: str,
                 output_activation_function_name: str,
                 is_relative: bool):
        super().__init__()
        self.norm_type = norm_type
        self.model_type = model_type
        self.in_keys = in_keys
        self.goal_latent_dim = goal_latent_dim
        self.state_dim = state_dim
        self.actions_dim = actions_dim
        self.action_space_low = action_space_low.to(device)
        self.action_space_high = action_space_high.to(device)
        self.tdm_planning_horizon_dim = 1
        self.input_dim = sum([get_dim(self, key) for key in in_keys])
        self.state_net = self.create_state_net(model_type, obs_dim, hidden_layers_out_features, goal_latent_dim, use_batch_norm, hidden_activation_function_name, output_activation_function_name).to(device)
        goal_key_index = in_keys.index('goal_latent')
        self.goal_latent_feature_index_start = sum([get_dim(self, key) for key in in_keys[:goal_key_index]])
        self.goal_latent_feature_index_end = self.goal_latent_feature_index_start + goal_latent_dim
        obs_key_index = in_keys.index('pixels_latent')
        self.obs_latent_feature_index_start = sum([get_dim(self, key) for key in in_keys[:obs_key_index]])
        self.obs_latent_feature_index_end = self.obs_latent_feature_index_start + goal_latent_dim
        self.is_relative = is_relative
    
    def create_state_net(self, model_type: str, obs_dim: int, hidden_layers_out_features: int, goal_latent_dim: int, use_batch_norm: bool, hidden_activation_function_name: str, output_activation_function_name: str):
        if model_type == "mini_resnet_3":
            last_out_channels = 512
            assert "pixels_latent" not in self.critic_in_keys, "MiniResNet3 is expected to receive the observation image, not its latent representation!"
            assert "pixels_transformed" in self.critic_in_keys, "MinResNet3 needs the pixels_transformed in_key, it works on the obs image directly!"
            fc1_in_features = last_out_channels + self.input_dim
            state_net = MiniResNet3(in_channels=obs_dim,
                                         fc1_in_features=fc1_in_features,
                                         fc1_out_features=hidden_layers_out_features[0],
                                         out_dim=goal_latent_dim)
        elif model_type == "mlp_pretrained_encoder":
            state_net = SimpleMlp(input_dim=self.input_dim,
                                  hidden_layers_out_features=hidden_layers_out_features,
                                  use_batch_norm=use_batch_norm,
                                  hidden_activation_function_name=hidden_activation_function_name,
                                  output_activation_function_name=output_activation_function_name,
                                  out_dim=goal_latent_dim)
        else:
            raise ValueError(f"Unknown model type '{model_type}'!")
        
        return state_net

    def forward(self, x: torch.Tensor, output_predicted_latent_state: bool = False) -> torch.Tensor:
        goal_latent = x[:, self.goal_latent_feature_index_start:self.goal_latent_feature_index_end]
        
        state_net_output = self.state_net(x)
        
        if self.is_relative:
            obs_latent = x[:, self.obs_latent_feature_index_start:self.obs_latent_feature_index_end]
            obs_latent = obs_latent + state_net_output
        else:
            obs_latent = state_net_output
        
        distance = compute_distance(norm_type=self.norm_type,
                                    obs_latent=obs_latent,
                                    goal_latent=goal_latent)
        
        q_value = -distance
        
        if output_predicted_latent_state:
            return q_value, obs_latent
        
        return q_value
    