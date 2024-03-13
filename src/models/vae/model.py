import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

    
class VAEModel(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 encoder_hidden_dims: list,
                 encoder_kernels: list,
                 encoder_strides: list,
                 encoder_paddings: list,
                 encoder_last_layer_fc: bool,
                 encoder_last_spatial_dim: int,
                 latent_dim: int,
                 decoder_hidden_dims: list,
                 decoder_kernels: list,
                 decoder_strides: list,
                 decoder_paddings: list):
        super().__init__()
        
        self.encoder = VAEEncoder(input_dim=input_dim,
                                  encoder_hidden_dims=encoder_hidden_dims,
                                  encoder_kernels=encoder_kernels,
                                  encoder_strides=encoder_strides,
                                  encoder_paddings=encoder_paddings,
                                  encoder_last_layer_fc=encoder_last_layer_fc,
                                  encoder_last_spatial_dim=encoder_last_spatial_dim,
                                  latent_dim=latent_dim)
        self.decoder = VAEDecoder(input_dim=input_dim,
                                  encoder_last_layer_fc=encoder_last_layer_fc,
                                  encoder_last_spatial_dim=encoder_last_spatial_dim,
                                  latent_dim=latent_dim,
                                  decoder_hidden_dims=decoder_hidden_dims,
                                  decoder_kernels=decoder_kernels,
                                  decoder_strides=decoder_strides,
                                  decoder_paddings=decoder_paddings)
        
    def forward(self, x) -> tuple[dist.Normal, dist.Normal]:
        q_z_given_x = self.encoder(x)
        
        z = q_z_given_x.rsample()
        
        p_x_given_Z = self.decoder(z)
        
        return q_z_given_x, p_x_given_Z


class VAEEncoder(nn.Module):
    def __init__(self,
                 input_dim: int, 
                 encoder_hidden_dims: list,
                 encoder_kernels: list,
                 encoder_strides: list,
                 encoder_paddings: list,
                 encoder_last_layer_fc: bool,
                 encoder_last_spatial_dim: int,
                 latent_dim: int):
        super().__init__()
        self.encoder_last_layer_fc=encoder_last_layer_fc
        self.hidden_layers = nn.ModuleList()
        
        for i, (out_channels, kernel, stride, padding) in enumerate(zip(encoder_hidden_dims, encoder_kernels, encoder_strides, encoder_paddings)):
            if i == 0:
                hidden_layer = nn.Conv2d(in_channels=input_dim, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding)
            else:
                hidden_layer = nn.Conv2d(in_channels=last_out_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding)  # noqa: F821
            
            self.hidden_layers.append(hidden_layer)
            last_out_channels = out_channels
        
        if encoder_last_layer_fc:
            in_features = last_out_channels * encoder_last_spatial_dim * encoder_last_spatial_dim
            self.mu = nn.Linear(in_features=in_features, out_features=latent_dim)
            self.log_var = nn.Linear(in_features=in_features, out_features=latent_dim)
        else:
            self.mu = nn.Conv2d(in_channels=last_out_channels, out_channels=latent_dim, kernel_size=1, stride=1, padding=0)
            self.log_var = nn.Conv2d(in_channels=last_out_channels, out_channels=latent_dim, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x: torch.Tensor) -> dist.Normal:
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        if self.encoder_last_layer_fc:
            x = torch.flatten(x, start_dim=1)

        mu = self.mu(x)
        
        log_var = self.log_var(x)
        std = (0.5 * log_var).exp()
        
        return dist.Normal(loc=mu, scale=std)


class VAEDecoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 encoder_last_layer_fc: int,
                 encoder_last_spatial_dim: int,
                 latent_dim: int,
                 decoder_hidden_dims: int,
                 decoder_kernels: int,
                 decoder_strides: int,
                 decoder_paddings: int):
        super().__init__()
        self.encoder_last_layer_fc = encoder_last_layer_fc
        self.encoder_last_spatial_dim = encoder_last_spatial_dim
        self.latent_dim = latent_dim
        self.hidden_layers = nn.ModuleList()
        
        for i, (hidden_dim, kernel, stride, padding) in enumerate(zip(decoder_hidden_dims, decoder_kernels, decoder_strides, decoder_paddings)):
            if i == 0:
                if encoder_last_layer_fc:
                    hidden_layer = nn.Linear(in_features=latent_dim, out_features=hidden_dim*encoder_last_spatial_dim*encoder_last_spatial_dim)
                else:
                    hidden_layer = nn.ConvTranspose2d(in_channels=latent_dim, out_channels=hidden_dim, kernel_size=kernel, stride=stride, padding=padding)
            else:
                hidden_layer = nn.ConvTranspose2d(in_channels=last_hidden_dim, out_channels=hidden_dim, kernel_size=kernel, stride=stride, padding=padding)  # noqa: F821
            
            self.hidden_layers.append(hidden_layer)
            last_hidden_dim = hidden_dim

        self.mu = nn.ConvTranspose2d(in_channels=last_hidden_dim, out_channels=input_dim, kernel_size=1, stride=1, padding=0)
    
    def forward(self, z: torch.Tensor) -> dist.Normal:        
        for i, hidden_layer in enumerate(self.hidden_layers):
            if i == 1 and self.encoder_last_layer_fc:
                batch_size = z.shape[0]
                z = z.reshape(batch_size, hidden_layer.in_channels, self.encoder_last_spatial_dim, self.encoder_last_spatial_dim)
                
            z = F.relu(hidden_layer(z))
        
        mu = self.mu(z)
                
        std = torch.ones_like(mu)
        
        return dist.Normal(loc=mu, scale=std)
