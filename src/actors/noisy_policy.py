import torch
from tensordict import TensorDict
from torchrl.modules.tensordict_module.exploration import TensorDictModuleWrapper
from torchrl.data.tensor_specs import CompositeSpec

class GaussianPolicy(TensorDictModuleWrapper):
    def __init__(self, 
                 policy, 
                 spec,
                 mean: float,
                 std: float,
                 clip: float,
                 action_key: str = "action"):
        super().__init__(policy)
        self.action_low = spec.space.low
        self.action_high = spec.space.high
        self.mean = mean
        self.std = std
        self.clip = clip
        self.action_key = action_key
        self.out_keys = list(self.td_module.out_keys)
        self.distribution = torch.distributions.Normal(loc=self.mean, scale=self.std)
        if action_key not in self.out_keys:
            raise RuntimeError(
                f"The action key {action_key} was not found in the td_module out_keys {self.td_module.out_keys}."
            )
        if spec is not None:
            if not isinstance(spec, CompositeSpec) and len(self.out_keys) >= 1:
                spec = CompositeSpec({action_key: spec}, shape=spec.shape[:-1])
            self._spec = spec
        elif hasattr(self.td_module, "_spec"):
            self._spec = self.td_module._spec.clone()
            if action_key not in self._spec.keys(True, True):
                self._spec[action_key] = None
        elif hasattr(self.td_module, "spec"):
            self._spec = self.td_module.spec.clone()
            if action_key not in self._spec.keys(True, True):
                self._spec[action_key] = None
        else:
            self._spec = CompositeSpec({key: None for key in policy.out_keys})
    
    @property
    def spec(self):
        return self._spec
    
    def forward(self, tensordict: TensorDict) -> TensorDict:
        tensordict = self.td_module.forward(tensordict)
        out = tensordict.get(self.action_key)
        out = self._add_noise(out)
        tensordict.set(self.action_key, out)
        return tensordict
    
    def _add_noise(self, action: torch.Tensor) -> torch.Tensor:
        noise = self.distribution.sample(sample_shape=action.shape).clip(min=-self.clip, max=self.clip)
        action = action + noise
        action = torch.clip(action, min=self.action_low, max=self.action_high)
        return action
        
