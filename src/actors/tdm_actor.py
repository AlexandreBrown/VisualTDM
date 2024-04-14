import torch.optim as optim
import torch
import torch.nn as nn
from models.resnets.mini_resnets import MiniResNet3
from tensordict import TensorDict


class TdmActor(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 actions_dim: int,
                 goal_latent_dim: int,
                 fc1_out_features: int,
                 device: torch.device,
                 learning_rate: float,
                 polyak_avg: float):
        super().__init__()
        last_out_channels = 512
        tau_dim = 1
        fc1_in_features = last_out_channels + goal_latent_dim + tau_dim
        self.mean_net = MiniResNet3(in_channels=obs_dim, 
                                    fc1_in_features=fc1_in_features, 
                                    fc1_out_features=fc1_out_features, 
                                    out_dim=actions_dim).to(device)
        self.learning_rate = learning_rate
        self.polyak_avg = polyak_avg
        self.optimizer = optim.Adam(self.mean_net.parameters(), lr=learning_rate)
    
    def update(self, train_data: TensorDict, critic, logs: dict):
        actions_y_hat = self(x=train_data['pixels_transformed'],
                             goal_latent=train_data['goal_latent'],
                             tau=train_data['planning_horizon'])

        q_values = critic(x=train_data['pixels_transformed'],
                          action=actions_y_hat,
                          goal_latent=train_data['goal_latent'],
                          tau=train_data['planning_horizon'])

        loss = -q_values.sum(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        logs['actor_loss'] = loss.item()
    
    def forward(self, x: torch.Tensor, goal_latent: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            goal_latent = goal_latent.unsqueeze(0)
            tau = tau.unsqueeze(0)
        
        additional_fc_features = torch.cat([goal_latent, tau], dim=1)
        
        return self.mean_net(x, additional_fc_features).squeeze(0)
