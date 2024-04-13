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
    
    def _inv_apply_transform(self, state: torch.Tensor) -> torch.Tensor:
        return self.goal_latent
    
    def _step(self, tensordict, next_tensordict):
        return next_tensordict.set(self.out_keys_inv[0], self.goal_latent)
    
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        self.goal_latent = self.compute_latent(tensordict_reset[self.in_keys_inv[0]])
        tensordict_reset[self.out_keys_inv[0]] = self.goal_latent
        return tensordict_reset
    
    def compute_latent(self, state: torch.Tensor):
        image = state.to(self.encoder_decoder_model.device)
        
        input = TensorDict(source={"image": image.unsqueeze(0)}, batch_size=[])
        
        latent_representation = self.encoder_decoder_model(input)['q_z'].loc.squeeze(0)
                
        return latent_representation
    
    def transform_observation_spec(self, observation_spec):
        observation_spec[self.out_keys_inv[0]] = UnboundedContinuousTensorSpec(
            shape=observation_spec.shape,
            device=observation_spec.device,
        )
        return observation_spec
