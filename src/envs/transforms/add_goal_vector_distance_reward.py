from tensordict import TensorDict
from torchrl.envs.transforms.transforms import Transform
from torchrl.data import UnboundedContinuousTensorSpec
from rewards.distance import compute_distance


class AddGoalVectorDistanceReward(Transform):
    def __init__(self, distance_type: str, reward_dim: int):
        super().__init__(in_keys=["pixels_latent", "goal_latent"], out_keys=["reward"])
        self.distance_type = distance_type
        self.reward_dim = reward_dim
    
    def _call(self, tensordict: TensorDict):
        reward = self.compute_reward(tensordict)
        tensordict[self.out_keys[0]] = reward
        return tensordict
    
    def compute_reward(self, tensordict: TensorDict):
        device = tensordict.device
        next_obs_latent = tensordict[self.in_keys[0]].to(device)
        goal_latent = tensordict[self.in_keys[1]].to(device)

        distance = compute_distance(self.distance_type, next_obs_latent, goal_latent)

        return -distance
    
    def transform_reward_spec(self, reward_spec):
        reward_spec[self.out_keys[0]] = UnboundedContinuousTensorSpec(
            shape=(self.reward_dim,),
            device=reward_spec.device,
        )
        
        return reward_spec
