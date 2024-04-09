from typing import Sequence
from tensordict import TensorDictBase
from torchrl.envs.transforms import Transform
from typing import Tuple


class StepPlanningHorizon(Transform):
    def __init__(self, in_keys: Sequence[str | Tuple[str, ...]] = None, out_keys: Sequence[str | Tuple[str, ...]] | None = None, in_keys_inv: Sequence[str | Tuple[str, ...]] | None = None, out_keys_inv: Sequence[str | Tuple[str, ...]] | None = None, max_planning_horizon: int = 15):
        super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)
        self.step = 0
        self.max_planning_horizon = max_planning_horizon
    
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict[self.out_keys] = self.max_planning_horizon - self.step 
        self.step += 1
        return tensordict
    
    def reset(self, tensordict):
        super().reset(tensordict)
        self.step = 0
