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
        p_x = data["p_x"]
                
        p_z = Normal(loc=torch.zeros_like(q_z.loc), scale=torch.ones_like(q_z.scale))
        kl_divergence_q_z =  kl_divergence(q_z, p_z).mean(dim=1)
        
        channels_dim = 1
        height_dim = 2
        width_dim = 3
        log_p_x_given_z = p_x.log_prob(data["pixels_transformed"]).mean(dim=[channels_dim,height_dim,width_dim])
        
        loss = -(log_p_x_given_z - self.beta * kl_divergence_q_z).mean()
        
        return TensorDict(
            source={
                "loss": loss,
                "mean_log_p_x_given_z": log_p_x_given_z.mean().item(),
                "mean_kl_divergence_q_z": kl_divergence_q_z.mean().item()
            },
            batch_size=[]
        )
