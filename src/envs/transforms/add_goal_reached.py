import torch
from tensordict import TensorDict
from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs.transforms.transforms import Transform
from torchrl.data import DiscreteTensorSpec, UnboundedContinuousTensorSpec
from rewards.distance import compute_distance


class AddGoalLatentReached(Transform):
    def __init__(self, goal_reached_epsilon: float, terminate_when_goal_reached: bool, distance_type: str):
        super().__init__(in_keys=["pixels_latent", "goal_latent", "planning_horizon"], out_keys=["done", "goal_latent_reached", f"goal_latent_{distance_type}_distance"])
        self.goal_reached_epsilon = goal_reached_epsilon
        self.terminate_when_goal_reached = terminate_when_goal_reached
        self.distance_type = distance_type
    
    def _call(self, tensordict: TensorDict):
        next_obs_latent = tensordict[self.in_keys[0]]
        goal_latent = tensordict[self.in_keys[1]]
        distance = compute_distance(distance_type=self.distance_type, obs_latent=next_obs_latent, goal_latent=goal_latent).mean()
        goal_not_reached = (distance > self.goal_reached_epsilon).type(torch.uint8)
        planning_horizon = tensordict[self.in_keys[2]]
        done = tensordict["done"].type(torch.uint8)
        done = 1 - (1 - done) * (planning_horizon != 0).type(torch.uint8)
        if self.terminate_when_goal_reached:
            done = 1 - (1 - done) * goal_not_reached
        tensordict[self.out_keys[0]] = done.type(torch.bool)
        tensordict[self.out_keys[1]] = (1 - goal_not_reached).type(torch.bool)
        tensordict[self.out_keys[2]] = distance.type(torch.float32)
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
