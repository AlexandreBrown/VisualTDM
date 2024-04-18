import torch
from tensordict import TensorDict
from torchrl.envs.transforms.transforms import Transform
from torchrl.data import UnboundedContinuousTensorSpec


class RemoveDataFromObservation(Transform):
    def __init__(self, index_to_remove_from_obs: list, original_obs_nb_dims: int):
        super().__init__(in_keys=["observation"], out_keys=["observation"])
        self.index_to_remove_from_obs = index_to_remove_from_obs
        self.new_env_obs_shape = original_obs_nb_dims - len(index_to_remove_from_obs)
    
    def _call(self, tensordict: TensorDict):
        
        observation = tensordict[self.in_keys[0]]
        
        keep_mask = torch.ones(observation.shape, dtype=torch.bool)
        keep_mask[self.index_to_remove_from_obs] = False
        
        updated_obs = observation[keep_mask]
        
        tensordict[self.out_keys[0]] = updated_obs
        
        return tensordict
    
    def _reset(
        self, 
        tensordict: TensorDict, 
        tensordict_reset: TensorDict
    ) -> TensorDict:
        return self._call(tensordict_reset)
    
    def transform_observation_spec(self, observation_spec):
        observation_spec[self.out_keys[0]] = UnboundedContinuousTensorSpec(
            shape=(self.new_env_obs_shape,),
            device=observation_spec.device,
            dtype=observation_spec[self.out_keys[0]].dtype
        )
        
        return observation_spec
