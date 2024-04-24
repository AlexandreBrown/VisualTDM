import torch
from tensordict import TensorDict
from torchrl.envs.transforms.transforms import Transform
from torchrl.data import BoundedTensorSpec


class AddTdmDone(Transform):
    def __init__(self, max_frames_per_traj: int, terminate_when_goal_reached: bool, goal_reached_epsilon):
        super().__init__(in_keys=[], out_keys=["done"])
        self.max_frames_per_traj = max_frames_per_traj
        self.terminate_when_goal_reached = terminate_when_goal_reached
        self.goal_reached_epsilon = goal_reached_epsilon
    
    def _call(self, tensordict: TensorDict):

        done = tensordict['done'].type(torch.int8)
        
        done = torch.ones_like(done) - (1 - done) * (tensordict['planning_horizon'] != 0.0).type(torch.int8)
        
        if self.terminate_when_goal_reached: 
            distance = torch.linalg.vector_norm(tensordict['goal_latent'] - tensordict['pixels_latent'], ord=2, dim=-1)
            done = torch.ones_like(done) - (1 - done) * (distance > self.goal_reached_epsilon).type(torch.int8)
        
        tensordict[self.out_keys[0]] = done.type(torch.bool)
        
        return tensordict
    
    def transform_done_spec(self, done_spec):
        done_spec[self.out_keys[0]] = BoundedTensorSpec(
            low=0,
            high=1,
            shape=(1,),
            device=done_spec.device,
            dtype=torch.bool
        )
        
        return done_spec
