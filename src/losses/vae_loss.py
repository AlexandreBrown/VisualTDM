import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.objectives import LossModule
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

class VAELoss(LossModule):
    def __init__(self, vae_model: TensorDictModule, beta: float, training_steps: int, annealing_strategy: str, annealing_cycles: int, annealing_ratio: float):
        super().__init__()
        self.convert_to_functional(
            vae_model,
            "vae_model",
            create_target_params=False,
        )
        self.vae_in_keys = vae_model.in_keys
        self.beta = beta
        
        self.kl_div_loss_weight = torch.ones(training_steps)
        
        if annealing_strategy == "cyclic_linear":
            self.kl_div_loss_weight = self.frange_cycle_linear(n_iter=training_steps, start=0.0, stop=1.0,  n_cycle=annealing_cycles, ratio=annealing_ratio)

    def forward(self, input: TensorDict) -> TensorDict:
        data = input.select(*self.vae_in_keys)
        
        with self.vae_model_params.to_module(self.vae_model):
            data = self.vae_model(data)
        
        q_z = data["q_z"]
        
        p_z = Normal(loc=torch.zeros_like(q_z.loc), scale=torch.ones_like(q_z.scale))
        
        kl_divergence_q_z =  kl_divergence(q_z, p_z).sum(dim=1)
        
        kl_div_loss = self.kl_div_loss_weight * self.beta * kl_divergence_q_z
        
        p_x = data["p_x"]
        
        x = data['pixels_transformed']
        
        reconstruction_loss = -p_x.log_prob(x).sum(dim=[1,2,3])
        
        loss = (reconstruction_loss + kl_div_loss).mean()
        
        return TensorDict(
            source={
                "loss": loss,
                "mean_reconstruction_loss": reconstruction_loss.mean().item(),
                "mean_kl_divergence_loss": kl_div_loss.mean().item()
            },
            batch_size=[]
        )

    def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=5, ratio=0.5):
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