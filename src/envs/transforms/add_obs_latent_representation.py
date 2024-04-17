from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs.transforms.transforms import Transform
from torchrl.data import UnboundedContinuousTensorSpec
from models.vae.utils import encode_to_latent_representation


class AddObsLatentRepresentation(Transform):
    def __init__(self, encoder: TensorDictModule, latent_dim: int):
        super().__init__(in_keys=["pixels_transformed"], out_keys=["pixels_latent"])
        self.encoder = encoder
        self.latent_dim = latent_dim
    
    def _call(self, tensordict: TensorDict):
        device = tensordict.device
        pixels_transformed = tensordict[self.in_keys[0]]
        next_obs_latent = encode_to_latent_representation(encoder=self.encoder,
                                                     image=pixels_transformed,
                                                     device=device)
        tensordict[self.out_keys[0]] = next_obs_latent
        return tensordict
    
    def _reset(
        self, 
        tensordict: TensorDict, 
        tensordict_reset: TensorDict
    ) -> TensorDict:
        return self._call(tensordict_reset)
    
    def transform_observation_spec(self, observation_spec):
        observation_spec[self.out_keys[0]] = UnboundedContinuousTensorSpec(
            shape=(self.latent_dim,),
            device=observation_spec.device,
        )
        return observation_spec
