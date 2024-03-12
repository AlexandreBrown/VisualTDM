import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.objectives import LossModule
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

class VAELoss(LossModule):
    def __init__(self, vae_model: TensorDictModule, beta: float):
        super().__init__()
        self.convert_to_functional(
            vae_model,
            "vae_model",
            create_target_params=False,
        )
        self.vae_in_keys = vae_model.in_keys
        self.beta = beta

    def forward(self, input: TensorDict) -> TensorDict:
        data = input.select(*self.vae_in_keys)
        
        with self.vae_model_params.to_module(self.vae_model):
            data = self.vae_model(data)
        
        q_z = data["q_z"]
        
        p_z = Normal(loc=torch.zeros_like(q_z.loc), scale=torch.ones_like(q_z.scale))
        
        kl_divergence_q_z =  kl_divergence(q_z, p_z).sum(dim=1)
        
        p_x = data["p_x"]
        
        x = data['pixels_transformed']
        
        reconstruction_loss = -p_x.log_prob(x).sum(dim=[1,2,3])
        
        loss = (self.beta * kl_divergence_q_z + reconstruction_loss).mean()
        
        return TensorDict(
            source={
                "loss": loss,
                "mean_reconstruction_loss": reconstruction_loss.mean().item(),
                "mean_kl_divergence_q_z": kl_divergence_q_z.mean().item()
            },
            batch_size=[]
        )
