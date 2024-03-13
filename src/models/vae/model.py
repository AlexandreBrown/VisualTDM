import torch
import torch.nn as nn
import torch.distributions as dist
from models.activation_parsing import get_activation
    
class VAEModel(nn.Module):
    def __init__(self, 
                 input_spatial_dim: int,
                 input_channels: int, 
                 encoder_hidden_dims: list,
                 encoder_hidden_activation: str,
                 encoder_hidden_kernels: list,
                 encoder_hidden_strides: list,
                 encoder_hidden_paddings: list,
                 encoder_use_batch_norm: bool,
                 encoder_leaky_relu_neg_slope: float,
                 latent_dim: int,
                 decoder_hidden_dims: list,
                 decoder_hidden_activation: str,
                 decoder_hidden_kernels: list,
                 decoder_hidden_strides: list,
                 decoder_hidden_paddings: list,
                 decoder_output_kernel: int,
                 decoder_output_stride: int,
                 decoder_output_padding: int,
                 decoder_use_batch_norm: bool):
        super().__init__()
        
        scale_factor = 2**(len(encoder_hidden_dims))
        encoder_conv_output_spatial_dim = input_spatial_dim // scale_factor
        encoder_conv_output_dim = (encoder_hidden_dims[-1])*encoder_conv_output_spatial_dim*encoder_conv_output_spatial_dim
        
        self.encoder = VAEEncoder(input_channels=input_channels,
                                  hidden_dims=encoder_hidden_dims,
                                  hidden_activation=encoder_hidden_activation,
                                  hidden_kernels=encoder_hidden_kernels,
                                  hidden_strides=encoder_hidden_strides,
                                  hidden_paddings=encoder_hidden_paddings,
                                  hidden_conv_output_dim=encoder_conv_output_dim,
                                  use_batch_norm=encoder_use_batch_norm,
                                  leaky_relu_neg_slope=encoder_leaky_relu_neg_slope,
                                  latent_dim=latent_dim)
        self.decoder = VAEDecoder(input_channels=input_channels,
                                  encoder_conv_output_dim=encoder_conv_output_dim,
                                  encoder_conv_output_spatial_dim=encoder_conv_output_spatial_dim,
                                  latent_dim=latent_dim,
                                  hidden_dims=decoder_hidden_dims,
                                  hidden_activation=decoder_hidden_activation,
                                  hidden_kernels=decoder_hidden_kernels,
                                  hidden_strides=decoder_hidden_strides,
                                  hidden_paddings=decoder_hidden_paddings,
                                  output_kernel=decoder_output_kernel,
                                  output_stride=decoder_output_stride,
                                  output_padding=decoder_output_padding,
                                  use_batch_norm=decoder_use_batch_norm)
        
    def forward(self, x) -> tuple[dist.Normal, dist.Normal]:
        q_z_given_x = self.encoder(x)
        
        z = q_z_given_x.rsample()
        
        p_x_given_Z = self.decoder(z)
        
        return q_z_given_x, p_x_given_Z


class VAEEncoder(nn.Module):
    def __init__(self,
                 input_channels: int, 
                 hidden_dims: list,
                 hidden_activation: str,
                 hidden_kernels: list,
                 hidden_strides: list,
                 hidden_paddings: list,
                 hidden_conv_output_dim: int,
                 use_batch_norm: bool,
                 leaky_relu_neg_slope: float,
                 latent_dim: int):
        super().__init__()
        self.hidden_layers_activation_fn = get_activation(hidden_activation, leaky_relu_negative_slope=leaky_relu_neg_slope)
        self.hidden_layers = nn.ModuleList()
        
        for i, (out_channels, kernel, stride, padding) in enumerate(zip(hidden_dims, hidden_kernels, hidden_strides, hidden_paddings)):
            if i == 0:
                hidden_layer = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding)
            else:
                hidden_layer = nn.Conv2d(in_channels=last_out_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding)  # noqa: F821
            
            if use_batch_norm:
                hidden_layer = nn.Sequential(hidden_layer, nn.BatchNorm2d(out_channels))
            
            self.hidden_layers.append(hidden_layer)
            last_out_channels = out_channels  # noqa
        
        self.mu = nn.Linear(in_features=hidden_conv_output_dim, out_features=latent_dim)
        self.log_var = nn.Linear(in_features=hidden_conv_output_dim, out_features=latent_dim)
        
    def forward(self, x: torch.Tensor) -> dist.Normal:
        for hidden_layer in self.hidden_layers:
             x = self.hidden_layers_activation_fn(hidden_layer(x))

        x = torch.flatten(x, start_dim=1)

        mu = self.mu(x)
        
        log_var = self.log_var(x)
        std = (0.5 * log_var).exp()
                
        return dist.Normal(loc=mu, scale=std)


class VAEDecoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 encoder_conv_output_dim: int,
                 encoder_conv_output_spatial_dim: int,
                 latent_dim: int,
                 hidden_dims: list,
                 hidden_activation: str,
                 hidden_kernels: list,
                 hidden_strides: list,
                 hidden_paddings: list,
                 output_kernel: int,
                 output_stride: int,
                 output_padding: int,
                 use_batch_norm: bool):
        super().__init__()
        self.encoder_conv_output_dim = encoder_conv_output_dim
        self.encoder_conv_output_spatial_dim = encoder_conv_output_spatial_dim
        self.latent_dim = latent_dim
        self.hidden_layers_activation_fn = get_activation(hidden_activation)
        self.hidden_layers = nn.ModuleList()
        
        first_layer = nn.Linear(in_features=latent_dim, out_features=encoder_conv_output_dim)
        self.hidden_layers.append(first_layer)
        
        in_channels_for_first_conv = encoder_conv_output_dim / (encoder_conv_output_spatial_dim*encoder_conv_output_spatial_dim)
        last_hidden_dim = int(in_channels_for_first_conv)
        
        for i, (hidden_dim, kernel, stride, padding) in enumerate(zip(hidden_dims, hidden_kernels, hidden_strides, hidden_paddings)):
            hidden_layer = nn.ConvTranspose2d(in_channels=last_hidden_dim, out_channels=hidden_dim, kernel_size=kernel, stride=stride, padding=padding)
            if use_batch_norm:
                hidden_layer = nn.Sequential(hidden_layer, nn.BatchNorm2d(hidden_dim))
            
            self.hidden_layers.append(hidden_layer)
            last_hidden_dim = hidden_dim

        self.mu = nn.ConvTranspose2d(in_channels=last_hidden_dim, out_channels=input_channels, kernel_size=output_kernel, stride=output_stride, padding=output_padding)
    
    def forward(self, z: torch.Tensor) -> dist.Normal:   
        for i, hidden_layer in enumerate(self.hidden_layers):
            z = self.hidden_layers_activation_fn(hidden_layer(z))
            if i == 0:
                batch_size = z.shape[0]
                z = z.reshape(batch_size, -1, self.encoder_conv_output_spatial_dim, self.encoder_conv_output_spatial_dim)
                
        mu = self.mu(z)
        
        std = torch.ones_like(mu)
        
        return dist.Normal(loc=mu, scale=std)
