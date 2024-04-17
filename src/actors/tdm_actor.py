import copy
import torch.optim as optim
import torch
import torch.nn as nn
from models.resnets.mini_resnets import MiniResNet3
from models.mlps.simple_mlp import SimpleMlp
from tensordict import TensorDict
from tensor_utils import get_tensor


class TdmActor(nn.Module):
    def __init__(self,
                 model_type: str,
                 obs_dim: int,
                 actions_dim: int,
                 goal_latent_dim: int,
                 hidden_layers_out_features: list,
                 hidden_activation_function_name: str,
                 output_activation_function_name: str,
                 device: torch.device,
                 learning_rate: float,
                 polyak_avg: float,
                 action_scale: float,
                 action_bias: float,
                 state_dim: int,
                 actor_in_keys: list,
                 critic_in_keys: list):
        super().__init__()
        tau_dim = 1
        if model_type == "mini_resnet_3":
            last_out_channels = 512
            fc1_in_features = last_out_channels + goal_latent_dim + tau_dim
            self.mean_net = MiniResNet3(in_channels=obs_dim,
                                         fc1_in_features=fc1_in_features,
                                         fc1_out_features=hidden_layers_out_features[0],
                                         out_dim=actions_dim)
        elif model_type == "mlp_pretrained_encoder":
            input_dim = goal_latent_dim + state_dim + goal_latent_dim + tau_dim
            self.mean_net = SimpleMlp(input_dim=input_dim,
                                hidden_layers_out_features=hidden_layers_out_features,
                                hidden_activation_function_name=hidden_activation_function_name,
                                output_activation_function_name=output_activation_function_name,
                                out_dim=actions_dim)
        else:
            raise ValueError(f"Unknown model type '{model_type}'!")
        self.mean_net = self.mean_net.to(device)
        self.learning_rate = learning_rate
        self.polyak_avg = polyak_avg
        self.optimizer = optim.Adam(self.mean_net.parameters(), lr=learning_rate)
        self.action_scale = action_scale.unsqueeze(0).to(device)
        self.action_bias = action_bias.unsqueeze(0).to(device)
        self.device = device
        self.actor_in_keys = actor_in_keys
        self.critic_in_keys = critic_in_keys
    
    def update(self, train_data: TensorDict, critic) -> dict:
        actor_inputs = get_tensor(train_data, self.actor_in_keys)
        policy_actions = self(actor_inputs)

        critics_train_data = copy.deepcopy(train_data)
        critics_train_data['action'] = policy_actions
        critic_inputs = get_tensor(critics_train_data, self.critic_in_keys)
        q_values = critic(critic_inputs)

        loss = -q_values.sum(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'actor_loss': loss.item()
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        x = x.to(self.device)
        self.mean_net = self.mean_net.to(self.device)
        
        output = self.mean_net(x) * self.action_scale + self.action_bias
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output
