import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

    
class VAEModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = VAEEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = VAEDecoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
        
    
    def forward(self, x) -> tuple[dist.Normal, dist.Normal]:
        q_Z_given_x = self.encoder(x)
        
        z = q_Z_given_x.rsample()
        
        p_x_given_Z = self.decoder(z)
        
        return q_Z_given_x, p_x_given_Z


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
    
    def forward(self, x) -> dist.Normal:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        mu = self.mu(x)
        log_var = self.log_var(x)
        std = (0.5 * log_var).exp()
        
        return dist.Normal(loc=mu, scale=std)


class VAEDecoder(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=latent_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=0)
        self.conv2 = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=0)
        self.conv4 = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1, padding=0)
        
        self.mu = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=1, kernel_size=5, stride=1, padding=0)
    
    def forward(self, x) -> dist.Normal:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        mu = self.mu(x)
        std = torch.ones_like(mu)
        
        return dist.Normal(loc=mu, scale=std)
