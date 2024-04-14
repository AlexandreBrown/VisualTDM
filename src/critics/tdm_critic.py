import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from tensordict import TensorDict
from actors.tdm_actor import TdmActor
from models.resnets.mini_resnets import MiniResNet3
from envs.transforms.compute_latent_goal_distance_vector_reward import ComputeLatentGoalDistanceVectorReward
from tensordict.nn import TensorDictModule

class TdmCritic(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 actions_dim: int,
                 goal_latent_dim: int,
                 fc1_out_features: int,
                 device: torch.device,
                 norm_type: str,
                 encoder: TensorDictModule,
                 actor: TdmActor,
                 learning_rate: float,
                 polyak_avg: float):
        super().__init__()
        last_out_channels = 512
        tau_dim = 1
        fc1_in_features = last_out_channels + actions_dim + goal_latent_dim + tau_dim
        self.q_fun = MiniResNet3(in_channels=obs_dim,
                                 fc1_in_features=fc1_in_features,
                                 fc1_out_features=fc1_out_features,
                                 out_dim=goal_latent_dim).to(device)
        self.q_fun_target = copy.deepcopy(self.q_fun).to(device)
        self.reward_transform = ComputeLatentGoalDistanceVectorReward(norm_type=norm_type, encoder=encoder, latent_dim=goal_latent_dim)
        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.learning_rate = learning_rate
        self.polyak_avg = polyak_avg
        self.optimizer = optim.Adam(self.q_fun.parameters(), lr=learning_rate)
    
    def update(self, train_data: TensorDict, logs: dict):
        y = self.compute_y(train_data)
        y_hat = self(x=train_data['pixels_transformed'],
                     action=train_data['action'],
                     goal_latent=train_data['goal_latent'],
                     tau=train_data['planning_horizon'])

        loss = F.mse_loss(y_hat, y, reduction='sum') / train_data.batch_size[0]
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        logs['critic_loss'] = loss.item()
    
    def compute_y(self, train_data: TensorDict) -> torch.Tensor:
        tau_0_reward = self.compute_tau_0_reward(train_data)
        
        planning_horizon = train_data['planning_horizon']
        tau_not_0_value = self.compute_tau_not_0_value(train_data, planning_horizon)
        
        y = tau_0_reward * (planning_horizon == 0) + tau_not_0_value * (planning_horizon != 0)
        y = y.detach()
        
        return y
    
    def compute_tau_0_reward(self, train_data: TensorDict) -> torch.Tensor:
        tau_0_reward_data = TensorDict(
            source={
                "pixels_transformed": train_data['next']['pixels_transformed'],
                "goal_latent": train_data['goal_latent']
            },
            batch_size=train_data.batch_size
        ).to(train_data.device)
        return self.reward_transform.compute_reward(tau_0_reward_data)
    
    def compute_tau_not_0_value(self, train_data: TensorDict, planning_horizon: torch.Tensor) -> torch.Tensor:
        actor_target_action = self.actor_target(x=train_data['next']['pixels_transformed'], goal_latent=train_data['goal_latent'], tau=planning_horizon - 1)
        
        additional_fc_features = torch.cat([actor_target_action, train_data['goal_latent'], planning_horizon - 1], dim=1)
        
        return self.q_fun_target(x=train_data['next']['pixels_transformed'], additional_fc_features=additional_fc_features)
    
    def forward(self, x: torch.Tensor, action: torch.Tensor, goal_latent: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        additional_fc_features = torch.cat([action, goal_latent, tau], dim=1)
        return self.q_fun(x=x, additional_fc_features=additional_fc_features)
    
    def update_target_network(self):
        for target_param, param in zip(self.q_fun_target.parameters(), self.q_fun.parameters()):
            target_param.data.copy_(self.polyak_avg * param.data + (1 - self.polyak_avg) * target_param.data)
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.polyak_avg * param.data + (1 - self.polyak_avg) * target_param.data)
