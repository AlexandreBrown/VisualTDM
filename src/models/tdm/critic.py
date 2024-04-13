import torch
import torch.nn as nn
from models.resnets.mini_resnets import MiniResNet3

class TdmCritic(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 actions_dim: int,
                 goal_latent_dim: int,
                 fc1_out_features: int,
                 device: torch.device):
        super().__init__()
        last_out_channels = 512
        tau_dim = 1
        fc1_in_features = last_out_channels + actions_dim + goal_latent_dim + tau_dim
        self.sate_net = MiniResNet3(in_channels=obs_dim,
                                    fc1_in_features=fc1_in_features,
                                    fc1_out_features=fc1_out_features,
                                    out_dim=goal_latent_dim).to(device)
        
    def forward(self, x: torch.Tensor, action: torch.Tensor, goal_latent: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        
        additional_fc_features = torch.cat([action, goal_latent, tau], dim=1)
        
        return self.sate_net(x, additional_fc_features)
