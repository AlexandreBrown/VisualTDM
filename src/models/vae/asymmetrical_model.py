import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

    
class VAEModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = VAEEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        encoder_last_conv_output_size = 58 # assumes input is 256x256
        self.decoder = VAEDecoder(encoder_input_dim=input_dim, encoder_hidden_dim=hidden_dim, encoder_latent_dim=latent_dim, encoder_last_conv_output_size=encoder_last_conv_output_size)
        
    def forward(self, x) -> tuple[dist.Normal, dist.Normal]:
        q_z_given_x = self.encoder(x)
        
        z = q_z_given_x.rsample()
        
        p_x_given_Z = self.decoder(z)
        
        return q_z_given_x, p_x_given_Z


class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=0)
        self.conv5 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=0)
        
        self.mu = nn.Conv2d(in_channels=hidden_dim, out_channels=latent_dim, kernel_size=4, stride=1, padding=0)
        self.log_var = nn.Conv2d(in_channels=hidden_dim, out_channels=latent_dim, kernel_size=4, stride=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> dist.Normal:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        mu = self.mu(x)
        mu = torch.flatten(mu, start_dim=1)
        
        log_var = self.log_var(x)
        std = (0.5 * log_var).exp()
        std = torch.flatten(std, start_dim=1)
        
        return dist.Normal(loc=mu, scale=std)


class VAEDecoder(nn.Module):
    def __init__(self, encoder_input_dim: int, encoder_hidden_dim: int, encoder_latent_dim: int, encoder_last_conv_output_size: int):
        super().__init__()
        self.encoder_latent_dim = encoder_latent_dim
        self.encoder_last_conv_output_size = encoder_last_conv_output_size
        self.conv1 = nn.ConvTranspose2d(in_channels=encoder_latent_dim, out_channels=encoder_hidden_dim, kernel_size=4, stride=1, padding=0)
        self.conv2 = nn.ConvTranspose2d(in_channels=encoder_hidden_dim, out_channels=encoder_hidden_dim, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.ConvTranspose2d(in_channels=encoder_hidden_dim, out_channels=encoder_hidden_dim, kernel_size=4, stride=2, padding=0)
        self.conv4 = nn.ConvTranspose2d(in_channels=encoder_hidden_dim, out_channels=encoder_hidden_dim, kernel_size=5, stride=1, padding=0)
        
        self.mu = nn.ConvTranspose2d(in_channels=encoder_hidden_dim, out_channels=encoder_input_dim, kernel_size=5, stride=1, padding=0)
    
    def forward(self, z: torch.Tensor) -> dist.Normal:
        batch_size = z.shape[0]
        z = z.reshape(batch_size, self.encoder_latent_dim, self.encoder_last_conv_output_size, self.encoder_last_conv_output_size)
        
        x = F.relu(self.conv1(z))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        mu = self.mu(x)
                
        std = torch.ones_like(mu)
        
        return dist.Normal(loc=mu, scale=std)
