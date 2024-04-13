import torch
import torch.nn as nn
from models.resnets.mini_resnets import MiniResNet3

class TdmPolicy(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 goal_latent_dim: int,
                 fc1_out_features: int,
                 actions_dim: int,
                 device: torch.device):
        super().__init__()
        last_out_channels = 512
        tau_dim = 1
        fc1_in_features = last_out_channels + goal_latent_dim + tau_dim
        self.mean_net = MiniResNet3(in_channels=obs_dim, 
                                    fc1_in_features=fc1_in_features, 
                                    fc1_out_features=fc1_out_features, 
                                    out_dim=actions_dim).to(device)
        
    def forward(self, x: torch.Tensor, goal_latent: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            goal_latent = goal_latent.unsqueeze(0)
            tau = tau.unsqueeze(0)
        
        additional_fc_features = torch.cat([goal_latent, tau], dim=1)
        
        return self.mean_net(x, additional_fc_features).squeeze(0)
