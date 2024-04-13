import torch
from torchrl.envs.transforms.transforms import Transform
from torchrl.data import UnboundedContinuousTensorSpec
from tensordict.nn import TensorDictModule
from tensordict import TensorDict, TensorDictBase


class AddGoalLatentRepresentation(Transform):
    def __init__(self, 
                 encoder_decoder_model: TensorDictModule, 
                 latent_dim: int):
        super().__init__(in_keys_inv=["goal_pixels_transformed"], 
                         out_keys_inv=["goal_latent"])
        self.encoder_decoder_model = encoder_decoder_model
        self.latent_dim = latent_dim
        
    def _inv_apply_transform(self, state: torch.Tensor):
        
        image = state.to(self.encoder_decoder_model.device)
        
        input = TensorDict(source={"image": image.unsqueeze(0)}, batch_size=[])
        
        latent_representation = self.encoder_decoder_model(input)['q_z'].loc.squeeze(0)
                
        return latent_representation
    
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        tensordict_reset = super()._reset(tensordict, tensordict_reset)
        goal_latent = self._inv_apply_transform(tensordict_reset[self.in_keys_inv[0]])
        tensordict_reset[self.out_keys_inv[0]] = goal_latent
        return tensordict_reset
    
    def transform_observation_spec(self, observation_spec):
        observation_spec[self.out_keys_inv[0]] = UnboundedContinuousTensorSpec(
            shape=(self.latent_dim),
            device=observation_spec.device,
        )
        return observation_spec
