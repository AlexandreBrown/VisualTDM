import torch.optim as optim
import torch
import torch.nn as nn
from models.resnets.mini_resnets import MiniResNet3
from models.mlps.mlp import Mlp
from tensordict import TensorDict


class TdmActor(nn.Module):
    def __init__(self,
                 model_type: str,
                 obs_dim: int,
                 actions_dim: int,
                 goal_latent_dim: int,
                 hidden_layers_out_features: list,
                 hidden_activation_function_name: str,
                 output_activation_function_name: str,
                 device: torch.device,
                 learning_rate: float,
                 polyak_avg: float,
                 action_scale: float,
                 action_bias: float):
        super().__init__()
        tau_dim = 1
        if model_type == "mini_resnet_3":
            last_out_channels = 512
            fc1_in_features = last_out_channels + goal_latent_dim + tau_dim
            self.mean_net = MiniResNet3(in_channels=obs_dim,
                                         fc1_in_features=fc1_in_features,
                                         fc1_out_features=hidden_layers_out_features[0],
                                         out_dim=actions_dim)
        elif model_type == "mlp_pretrained_encoder":
            input_dim = goal_latent_dim + goal_latent_dim + tau_dim
            self.mean_net = Mlp(input_dim=input_dim,
                                hidden_layers_out_features=hidden_layers_out_features,
                                hidden_activation_function_name=hidden_activation_function_name,
                                output_activation_function_name=output_activation_function_name,
                                out_dim=actions_dim)
        else:
            raise ValueError(f"Unknown model type '{model_type}'!")
        self.mean_net = self.mean_net.to(device)
        self.learning_rate = learning_rate
        self.polyak_avg = polyak_avg
        self.optimizer = optim.Adam(self.mean_net.parameters(), lr=learning_rate)
        self.action_scale = action_scale.unsqueeze(0).to(device)
        self.action_bias = action_bias.unsqueeze(0).to(device)
        self.device = device
    
    def update(self, train_data: TensorDict, critic) -> dict:
        policy_actions = self(x=train_data['pixels_latent'],
                              goal_latent=train_data['goal_latent'],
                              tau=train_data['planning_horizon'])

        q_values = critic(x=train_data['pixels_latent'],
                          action=policy_actions,
                          goal_latent=train_data['goal_latent'],
                          tau=train_data['planning_horizon'])

        loss = -q_values.sum(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'actor_loss': loss.item()
        }

    def forward(self, x: torch.Tensor, goal_latent: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            goal_latent = goal_latent.unsqueeze(0)
            tau = tau.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        x = x.to(self.device)
        additional_fc_features = torch.cat([goal_latent, tau], dim=1).to(self.device)
        self.mean_net = self.mean_net.to(self.device)
        
        output = self.mean_net(x, additional_fc_features) * self.action_scale + self.action_bias
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output
