import torch
from tensordict import TensorDict
from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs.transforms.transforms import Transform
from torchrl.data import DiscreteTensorSpec, UnboundedContinuousTensorSpec


class AddGoalReached(Transform):
    def __init__(self, goal_reached_epsilon: float):
        super().__init__(in_keys=["pixels_latent", "goal_latent", "planning_horizon"], out_keys=["done", "goal_reached", "goal_l2_distance"])
        self.goal_reached_epsilon = goal_reached_epsilon
    
    def _call(self, tensordict: TensorDict):
        next_obs_latent = tensordict[self.in_keys[0]]
        goal_latent = tensordict[self.in_keys[1]]
        diff = next_obs_latent - goal_latent
        norm = torch.norm(diff, dim=0, keepdim=True)
        goal_not_reached = (norm > self.goal_reached_epsilon).type(torch.uint8)
        done = tensordict[self.out_keys[0]].type(torch.uint8)
        planning_horizon = tensordict[self.in_keys[2]]
        done = 1 - (1 - done) * (planning_horizon != 0).type(torch.uint8)
        done = 1 - (1 - done) * goal_not_reached
        tensordict[self.out_keys[0]] = done.type(torch.bool)
        tensordict[self.out_keys[1]] = (1 - goal_not_reached).type(torch.bool)
        tensordict[self.out_keys[2]] = norm.type(torch.float32)
        return tensordict

    def _reset(
        self, 
        tensordict: TensorDict, 
        tensordict_reset: TensorDict
    ) -> TensorDict:
        return self._call(tensordict_reset)
    
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        observation_spec[self.out_keys[1]] = DiscreteTensorSpec(
            n=2,
            shape=(1,),
            device=observation_spec.device,
            dtype=torch.bool
        )
        observation_spec[self.out_keys[2]] = UnboundedContinuousTensorSpec(
            shape=(1,),
            device=observation_spec.device,
            dtype=torch.float32
        )
        return observation_spec
    
    def transform_done_spec(self, done_spec):
        done_spec[self.out_keys[0]] = DiscreteTensorSpec(
            n=done_spec[self.out_keys[0]].n,
            shape=done_spec[self.out_keys[0]].shape,
            device=done_spec[self.out_keys[0]].device,
            dtype=done_spec[self.out_keys[0]].dtype
        )
        
        return done_spec
