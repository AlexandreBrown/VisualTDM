import torch.nn as nn
import torch
from tensordict.nn import TensorDictModule
from models.activation_parsing import get_activation
from models.vae.utils import encode_to_latent_representation


class MlpPretrainedEncoder(nn.Module):
    def __init__(self, 
                 encoder: TensorDictModule,
                 input_dim: int,
                 hidden_layers_out_features: list, 
                 hidden_activation_function_name: str, 
                 output_activation_function_name: str,
                 out_dim: int):
        super().__init__()
        
        self.encoder = encoder
        
        in_features = input_dim
        self.hidden_layers = nn.ModuleList()
        for hidden_layer_out_features in hidden_layers_out_features:
            self.hidden_layers.append(nn.Linear(in_features=in_features, out_features=hidden_layer_out_features))
            in_features = hidden_layer_out_features
        
        self.hidden_activation_function_name = hidden_activation_function_name
        self.output_activation_function_name = output_activation_function_name
        
        self.output_layer = nn.Linear(in_features=in_features, out_features=out_dim)
    
    def forward(self, x: torch.Tensor, additional_fc_features: torch.Tensor) -> torch.Tensor:
        device = x.device
        x = encode_to_latent_representation(encoder=self.encoder,
                                            image=x,
                                            device=device)
        
        x = torch.cat([x, additional_fc_features], dim=1)
        
        for hidden_layer in self.hidden_layers:
            activation_function = get_activation(self.hidden_activation_function_name)
            x = activation_function(hidden_layer(x))
        
        activation_function = get_activation(self.output_activation_function_name)
        
        return activation_function(self.output_layer(x))
