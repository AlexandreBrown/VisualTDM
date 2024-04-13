import torch
from tensordict import TensorDictBase
from torchrl.envs.transforms.transforms import Transform
from torchrl.data import BoundedTensorSpec


class AddPlanningHorizon(Transform):
    def __init__(self, initial_max_planning_horizon: int):
        super().__init__(in_keys=[], out_keys=["planning_horizon"], in_keys_inv=None, out_keys_inv=None)
        self.planning_horizon = float(initial_max_planning_horizon)
        self.max_planning_horizon = float(initial_max_planning_horizon)
        
    def _call(self, tensordict: TensorDictBase):
        
        if self.planning_horizon == 0.:
            self.planning_horizon = self.max_planning_horizon
        
        value = torch.full(size=(1,), fill_value=self.planning_horizon, device=tensordict.device)
        tensordict[self.out_keys[0]] = value
        
        self.planning_horizon -= 1.
        
        return tensordict
    
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        self.planning_horizon = self.max_planning_horizon
        return self._call(tensordict_reset)
    
    def transform_observation_spec(self, observation_spec):
        observation_spec[self.out_keys[0]] = BoundedTensorSpec(
            low=1,
            high=self.max_planning_horizon,
            shape=observation_spec.shape,
            device=observation_spec.device,
        )
        
        return observation_spec