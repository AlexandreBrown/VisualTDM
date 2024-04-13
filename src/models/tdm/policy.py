import torch
import torch.nn as nn
from models.resnets.mini_resnets import MiniResNet3

class TdmPolicy(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 goal_dim: int,
                 fc1_out_features: int,
                 actions_dim: int,
                 device: torch.device):
        super().__init__()
        self.goal_dim = goal_dim
        self.mean_net = MiniResNet3(in_channels=obs_dim, goal_dim=goal_dim, fc1_out_features=fc1_out_features, out_dim=actions_dim).to(device)
        
    def forward(self, x: torch.Tensor, goal_latent: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            goal_latent = goal_latent.unsqueeze(0)
            tau = tau.unsqueeze(0)
        
        return self.mean_net(x, goal_latent, tau).squeeze(0)
