import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

    
class VAEModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        encoder_dim_before_fc = 29 # assumes input is 128x128
        
        self.encoder = VAEEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, dim_before_fc=encoder_dim_before_fc)
        self.decoder = VAEDecoder(encoder_input_dim=input_dim, encoder_hidden_dim=hidden_dim, encoder_latent_dim=latent_dim, encoder_dim_before_fc=encoder_dim_before_fc)
        
    def forward(self, x) -> tuple[dist.Normal, dist.Normal]:
        q_z_given_x = self.encoder(x)
        
        z = q_z_given_x.rsample()
        
        p_x_given_Z = self.decoder(z)
        
        return q_z_given_x, p_x_given_Z


class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, encoder_dim_before_fc: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=0)
        self.conv5 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=0)
        
        in_features = hidden_dim * encoder_dim_before_fc * encoder_dim_before_fc
        self.mu = nn.Linear(in_features=in_features, out_features=latent_dim)
        self.log_var = nn.Linear(in_features=in_features, out_features=latent_dim)
    
    def forward(self, x: torch.Tensor) -> dist.Normal:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        x = torch.flatten(x, start_dim=1)
                
        mu = self.mu(x)
        
        log_var = self.log_var(x)
        std = (0.5 * log_var).exp()
        
        return dist.Normal(loc=mu, scale=std)


class VAEDecoder(nn.Module):
    def __init__(self, encoder_input_dim: int, encoder_hidden_dim: int, encoder_latent_dim: int, encoder_dim_before_fc: int):
        super().__init__()
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_dim_before_fc = encoder_dim_before_fc
        self.fc1 = nn.Linear(in_features=encoder_latent_dim, out_features=encoder_hidden_dim * encoder_dim_before_fc * encoder_dim_before_fc)
        self.conv1 = nn.ConvTranspose2d(in_channels=encoder_hidden_dim, out_channels=encoder_hidden_dim, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.ConvTranspose2d(in_channels=encoder_hidden_dim, out_channels=encoder_hidden_dim, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.ConvTranspose2d(in_channels=encoder_hidden_dim, out_channels=encoder_hidden_dim, kernel_size=5, stride=1, padding=0)
        
        self.mu = nn.ConvTranspose2d(in_channels=encoder_hidden_dim, out_channels=encoder_input_dim, kernel_size=5, stride=1, padding=0)
    
    def forward(self, z: torch.Tensor) -> dist.Normal:
        batch_size = z.shape[0]
        
        x = F.relu(self.fc1(z))
        x = x.reshape(batch_size, self.encoder_hidden_dim, self.encoder_dim_before_fc, self.encoder_dim_before_fc)
        
        x = F.relu(self.conv1(x))        
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        mu = self.mu(x)
                
        std = torch.ones_like(mu)
        
        return dist.Normal(loc=mu, scale=std)
