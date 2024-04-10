import torch
import torch.nn as nn
from models.resnets.mini_resnets import MiniResNet3

class TdmQFunction(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 goal_dim: int,
                 fc1_out_features: int,
                 device: torch.device):
        super().__init__()
        self.sate_net = MiniResNet3(in_channels=obs_dim, goal_dim=goal_dim, fc1_out_features=fc1_out_features, out_dim=goal_dim).to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        pass
