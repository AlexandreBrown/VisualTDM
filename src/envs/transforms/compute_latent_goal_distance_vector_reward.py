
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.envs.transforms.transforms import Transform
from torchrl.data import UnboundedContinuousTensorSpec
from rewards.distance import compute_distance
from models.vae.utils import encode_to_latent_representation


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
        device = tensordict.device
        self.encoder = self.encoder.to(device)
        pixels_transformed = tensordict[self.in_keys[0]]
        obs_latent = encode_to_latent_representation(encoder=self.encoder,
                                                     image=pixels_transformed,
                                                     device=device)
        goal_latent = tensordict["goal_latent"].to(device)

        distance = compute_distance(self.norm_type, obs_latent, goal_latent)

        return -distance
    
    def transform_reward_spec(self, reward_spec):
        reward_spec[self.out_keys[0]] = UnboundedContinuousTensorSpec(
            shape=(self.latent_dim,),
            device=reward_spec.device,
        )
        
        return reward_spec
