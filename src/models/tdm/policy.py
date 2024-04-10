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
        
    def forward(self, x: torch.Tensor, goal: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        if len(tau.shape) == 0:
            tau = tau.unsqueeze(0).unsqueeze(1)
        
        if goal is None:
            goal = torch.randn(x.shape[0], self.goal_dim)
        
        return self.mean_net(x, goal, tau).squeeze(0)
