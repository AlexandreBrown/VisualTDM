import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from tensordict import TensorDict
from actors.tdm_actor import TdmActor
from models.resnets.mini_resnets import MiniResNet3
from models.mlps.mlp_encoder_pretrained import MlpPretrainedEncoder
from tensordict.nn import TensorDictModule
from models.vae.utils import encode_to_latent_representation
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
                 encoder: TensorDictModule,
                 actor: TdmActor,
                 learning_rate: float,
                 polyak_avg: float):
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
            input_dim = goal_latent_dim + actions_dim + goal_latent_dim + tau_dim
            self.state_net = MlpPretrainedEncoder(encoder=encoder,
                                       input_dim=input_dim,
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
        self.encoder = encoder
        self.device = device
    
    def update(self, train_data: TensorDict) -> dict:
        y = self.compute_y(train_data)
        y_hat = self(x=train_data['pixels_transformed'],
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
    
    def compute_y(self, train_data: TensorDict) -> torch.Tensor:
        tau_0_reward = self.compute_tau_0_reward(train_data)
        
        planning_horizon = train_data['planning_horizon']
        tau_not_0_reward = self.compute_tau_not_0_reward(train_data, planning_horizon)
        
        y = tau_0_reward * (planning_horizon == 0) + tau_not_0_reward * (planning_horizon != 0)
        y = y.detach()
        
        return y
    
    def compute_tau_0_reward(self, train_data: TensorDict) -> torch.Tensor:
        obs_latent = encode_to_latent_representation(encoder=self.encoder,
                                                     image=train_data['next']['pixels_transformed'],
                                                     device=train_data.device)
        
        goal_latent = train_data['goal_latent']
        
        distance = compute_distance(norm_type=self.norm_type,
                                    obs_latent=obs_latent,
                                    goal_latent=goal_latent)
        
        return -distance
    
    def compute_tau_not_0_reward(self, train_data: TensorDict, planning_horizon: torch.Tensor) -> torch.Tensor:
        goal_latent = train_data['goal_latent']
        actor_target_action = self.actor_target(x=train_data['next']['pixels_transformed'], goal_latent=goal_latent, tau=planning_horizon - 1)
        
        additional_fc_features = torch.cat([actor_target_action, goal_latent, planning_horizon - 1], dim=1)
        obs_latent = self.state_net_target(x=train_data['next']['pixels_transformed'], additional_fc_features=additional_fc_features)
        
        distance = compute_distance(norm_type=self.norm_type,
                                    obs_latent=obs_latent,
                                    goal_latent=goal_latent)

        return -distance
    
    def forward(self, x: torch.Tensor, action: torch.Tensor, goal_latent: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        additional_fc_features = torch.cat([action, goal_latent, tau], dim=1)
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
