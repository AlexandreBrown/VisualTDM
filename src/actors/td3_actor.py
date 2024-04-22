import torch.optim as optim
import torch
import torch.nn as nn
from models.resnets.mini_resnets import MiniResNet3
from models.mlps.simple_mlp import SimpleMlp
from tensordict import TensorDict
from tensor_utils import get_tensor
from envs.dimensions import get_dim


class Td3Actor(nn.Module):
    def __init__(self,
                 model_type: str,
                 actions_dim: int,
                 goal_dim: int,
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
                 critic_in_keys: list,
                 obs_dim: int = None):
        super().__init__()
        self.goal_dim = goal_dim
        self.state_dim = state_dim
        self.actions_dim = actions_dim
        self.tdm_planning_horizon_dim = 1
        self.actor_input_dim = sum([get_dim(self, key) for key in actor_in_keys])
        if model_type == "mini_resnet_3":
            last_out_channels = 512
            fc1_in_features = last_out_channels + self.actor_input_dim
            self.mean_net = MiniResNet3(in_channels=obs_dim,
                                         fc1_in_features=fc1_in_features,
                                         fc1_out_features=hidden_layers_out_features[0],
                                         out_dim=actions_dim)
        elif model_type == "mlp":
            self.mean_net = SimpleMlp(input_dim=self.actor_input_dim,
                                hidden_layers_out_features=hidden_layers_out_features,
                                use_batch_norm=False,
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
        policy_actions = self(actor_inputs, train=True)

        critics_train_data = train_data.clone(recurse=True)
        critics_train_data['action'] = policy_actions
        critic_inputs = get_tensor(critics_train_data, self.critic_in_keys)
        q_values = critic(critic_inputs)

        loss = -q_values.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'actor_loss': loss.item()
        }

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        if not train:
            self.eval()
            self.mean_net.eval()
            with torch.no_grad():
                output = self.get_action_mean(x)
        else:
            self.train()
            self.mean_net.train()
            output = self.get_action_mean(x)
        
        return output
    
    def get_action_mean(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        mean = self.mean_net(x)
        output = mean * self.action_scale + self.action_bias
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output
