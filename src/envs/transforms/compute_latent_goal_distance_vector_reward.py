
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.envs.transforms.transforms import Transform
from torchrl.data import UnboundedContinuousTensorSpec


class ComputeLatentGoalDistanceVectorReward(Transform):
    def __init__(self, norm_type: str, encoder: TensorDictModule, latent_dim: int):
        super().__init__(in_keys=["pixels_transformed"], out_keys=["reward"])
        self.norm_type = norm_type
        self.encoder = encoder
        self.latent_dim = latent_dim
    
    def _call(self, tensordict: TensorDictBase):
        reward = self.compute_reward(tensordict)
        tensordict[self.out_keys[0]] = reward
        return tensordict
    
    def compute_reward(self, tensordict: TensorDict):
        pixels_transformed = tensordict[self.in_keys[0]]
        obs_latent = self.encoder_to_latent_representation(image=pixels_transformed, batch_size=tensordict.batch_size)
        goal_latent = tensordict["goal_latent"]

        if self.norm_type == 'l1':
            reward = torch.abs(obs_latent - goal_latent)
        elif self.norm_type == 'l2':
            reward = torch.sqrt(torch.pow(obs_latent - goal_latent, exponent=2))
        else:
            raise ValueError(f"Unknown goal norm type '{self.norm_type}'")

        return reward
  
    def encoder_to_latent_representation(self, image: torch.Tensor, batch_size: int) -> torch.Tensor:
        input = TensorDict(
            source={
                "image": image.unsqueeze(0).to(self.encoder.device)
            },
            batch_size=batch_size
        )
        return self.encoder(input)['q_z'].loc.squeeze(0)
    
    def transform_reward_spec(self, reward_spec):
        reward_spec[self.out_keys[0]] = UnboundedContinuousTensorSpec(
            shape=(self.latent_dim,),
            device=reward_spec.device,
        )
        
        return reward_spec
