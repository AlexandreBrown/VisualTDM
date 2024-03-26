import torch
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.objectives import LossModule
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

class VAELoss(LossModule):
    def __init__(self, 
                 vae_model: TensorDictModule, 
                 beta: float, 
                 training_steps: int, 
                 annealing_strategy: str, 
                 annealing_cycles: int, 
                 annealing_ratio: float, 
                 reconstruction_loss: str):
        super().__init__()
        self.convert_to_functional(
            vae_model,
            "vae_model",
            create_target_params=False,
        )
        self.vae_in_keys = vae_model.in_keys
        self.beta = beta
        
        if annealing_strategy == "cyclic_linear":
            self.kl_div_loss_weight = self.frange_cycle_linear(n_iter=training_steps, start=0.0, stop=1.0,  n_cycle=annealing_cycles, ratio=annealing_ratio)
        elif annealing_strategy == "linear":
            self.kl_div_loss_weight = self.frange_cycle_linear(n_iter=training_steps, start=0.0, stop=1.0,  n_cycle=1, ratio=annealing_ratio)
        else:
            self.kl_div_loss_weight = torch.ones(training_steps)
        
        self.reconstruction_loss_type = reconstruction_loss
        
    def forward(self, input: TensorDict, train_step: int) -> TensorDict:
        data = input.select(*self.vae_in_keys)
        
        with self.vae_model_params.to_module(self.vae_model):
            data = self.vae_model(data)
        
        q_z = data["q_z"]
        
        p_z = Normal(loc=torch.zeros_like(q_z.loc), scale=torch.ones_like(q_z.scale))
        
        kl_divergence_q_z =  kl_divergence(q_z, p_z).sum(dim=1).mean()
        
        kl_loss_weight = self.kl_div_loss_weight[train_step]
        
        kl_div_loss_no_annealing = self.beta * kl_divergence_q_z
        kl_div_loss = kl_loss_weight * kl_div_loss_no_annealing
        
        p_x = data["p_x"]
        
        x = data['pixels_transformed']
        
        batch_size = x.shape[0]
        if self.reconstruction_loss_type == 'mse':
            x_reconstructed = F.sigmoid(p_x.loc)
            reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction='sum') / batch_size
        elif self.reconstruction_loss_type == 'bce':
            x_reconstructed = F.sigmoid(p_x.loc)
            reconstruction_loss = F.binary_cross_entropy(
                x_reconstructed.reshape(batch_size, -1),
                x.reshape(batch_size, -1),
                reduction='sum',
            ) / batch_size
        elif self.reconstruction_loss_type == 'mean_logprob':
            reconstruction_loss = -p_x.log_prob(x).sum(dim=[1,2,3]).mean()
        else:
            raise ValueError(f"Unknown reconstruction loss '{reconstruction_loss}'")
                    
        loss = reconstruction_loss + kl_div_loss
        loss_no_annealing = reconstruction_loss + kl_div_loss_no_annealing
        return TensorDict(
            source={
                "loss": loss,
                "loss_no_annealing": loss_no_annealing.detach().cpu().item(),
                "mean_reconstruction_loss": reconstruction_loss.detach().cpu().mean().item(),
                "mean_kl_divergence_loss": kl_div_loss.detach().cpu().mean().item(),
                "mean_kl_divergence": kl_divergence_q_z.detach().cpu().mean().item(),
                "kl_loss_weight": kl_loss_weight.item()
            },
            batch_size=[]
        )

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=5, ratio=0.5):
        """
        Source: https://arxiv.org/abs/1903.10145
        """
        L = torch.ones(n_iter) * stop
        period = n_iter/n_cycle
        step = (stop-start)/(period*ratio)

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i+c*period) < n_iter):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L 