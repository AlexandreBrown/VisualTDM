import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from tensordict import TensorDict
from actors.tdm_actor import TdmActor
from models.resnets.mini_resnets import MiniResNet3
from models.mlps.mlp import Mlp
from rewards.distance import compute_distance

class TdmCritic(nn.Module):
    def __init__(self,
                 model_type: str,
                 obs_dim: int,
                 actions_dim: int,
                 goal_latent_dim: int,
                 hidden_layers_out_features: list,
                 hidden_activation_function_name: str,
                 output_activation_function_name: str,
                 device: torch.device,
                 norm_type: str,
                 actor: TdmActor,
                 learning_rate: float,
                 polyak_avg: float,
                 target_policy_action_clip: float,
                 state_dim: int):
        super().__init__()
        tau_dim = 1
        if model_type == "mini_resnet_3":
            last_out_channels = 512
            fc1_in_features = last_out_channels + actions_dim + goal_latent_dim + tau_dim
            self.state_net = MiniResNet3(in_channels=obs_dim,
                                         fc1_in_features=fc1_in_features,
                                         fc1_out_features=hidden_layers_out_features[0],
                                         out_dim=goal_latent_dim)
        elif model_type == "mlp_pretrained_encoder":
            input_dim = goal_latent_dim + state_dim + actions_dim + goal_latent_dim + tau_dim
            self.state_net = Mlp(input_dim=input_dim,
                                 hidden_layers_out_features=hidden_layers_out_features,
                                 hidden_activation_function_name=hidden_activation_function_name,
                                 output_activation_function_name=output_activation_function_name,
                                 out_dim=goal_latent_dim)
        else:
            raise ValueError(f"Unknown model type '{model_type}'!")
        self.state_net = self.state_net.to(device)
        self.state_net_target = copy.deepcopy(self.state_net).to(device)
        self.norm_type = norm_type
        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.learning_rate = learning_rate
        self.polyak_avg = polyak_avg
        self.optimizer = optim.Adam(self.state_net.parameters(), lr=learning_rate)
        self.device = device
        self.target_policy_action_clip = target_policy_action_clip
    
    def update(self, train_data: TensorDict) -> dict:
        y = self.compute_target(train_data)
        y_hat = self(x=train_data['pixels_latent'],
                     state=train_data['state'],
                     action=train_data['action'],
                     goal_latent=train_data['goal_latent'],
                     tau=train_data['planning_horizon'])

        loss = F.mse_loss(y_hat, y, reduction='sum') / train_data.batch_size[0]
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'critic_loss':loss.item()
        }
    
    def compute_target(self, train_data: TensorDict) -> torch.Tensor:
        
        next_pixels_latent = train_data['next']['pixels_latent']
        next_state = train_data['next']['state']
        goal_latent = train_data['goal_latent']
        planning_horizon = train_data['planning_horizon']
        rewards = train_data['next']['reward']
        done = train_data['next']['done'].type(torch.uint8)
        
        next_actions = self.actor_target(x=next_pixels_latent,
                                         state=next_state,
                                         goal_latent=goal_latent,
                                         tau=planning_horizon - 1
                                         ).clip(min=-self.target_policy_action_clip, max=self.target_policy_action_clip)
        
        target_additional_features = torch.cat([next_state, next_actions, goal_latent, planning_horizon - 1], dim=1)
        target_q_values = self.state_net_target(x=next_pixels_latent,
                                                additional_fc_features=target_additional_features)
        
        target = rewards + (1. - done) * target_q_values
        target = target.detach()
        
        return target
    
    def forward(self, x: torch.Tensor, state: torch.Tensor, action: torch.Tensor, goal_latent: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        additional_fc_features = torch.cat([state, action, goal_latent, tau], dim=1)
        obs_latent = self.state_net(x=x, additional_fc_features=additional_fc_features)
        
        distance = compute_distance(norm_type=self.norm_type,
                                    obs_latent=obs_latent,
                                    goal_latent=goal_latent)
        
        return -distance
    
    def update_target_network(self):
        for target_param, param in zip(self.state_net_target.parameters(), self.state_net.parameters()):
            target_param.data.copy_(self.polyak_avg * param.data + (1 - self.polyak_avg) * target_param.data)
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.polyak_avg * param.data + (1 - self.polyak_avg) * target_param.data)
