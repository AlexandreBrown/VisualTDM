import torch
import torch.nn as nn
from models.resnets.mini_resnets import MiniResNet3

class TdmCritic(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 goal_dim: int,
                 fc1_out_features: int,
                 device: torch.device):
        super().__init__()
        self.sate_net = MiniResNet3(in_channels=obs_dim, goal_dim=goal_dim, fc1_out_features=fc1_out_features, out_dim=goal_dim).to(device)
        
    def forward(self, x: torch.Tensor, goal_latent: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.sate_net(x, goal_latent, tau)
