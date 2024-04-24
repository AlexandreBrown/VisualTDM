import torch
from tensordict import TensorDict
from torchrl.envs.transforms.transforms import Transform
from torchrl.data import BoundedTensorSpec
from loggers.metrics.goal_reached_metric import GoalReachedMetric


class AddTdmDone(Transform):
    def __init__(self, terminate_when_goal_reached: bool, goal_latent_reached_metric: GoalReachedMetric):
        super().__init__(in_keys=[], out_keys=["done"])
        self.terminate_when_goal_reached = terminate_when_goal_reached
        self.goal_latent_reached_metric = goal_latent_reached_metric
    
    def _step(
        self, 
        tensordict: TensorDict, 
        next_tensordict: TensorDict
    ) -> TensorDict:
        done = next_tensordict['done'].type(torch.int8)
        
        done = torch.ones_like(done) - (1 - done) * (next_tensordict['planning_horizon'] != 0.0).type(torch.int8)
        
        if self.terminate_when_goal_reached:
            goal_reached = self.goal_latent_reached_metric.compute(TensorDict(source={
                'goal_latent': tensordict['goal_latent'],
                'next': {
                    'pixels_latent': next_tensordict['pixels_latent']
                }
            },batch_size=[]))
            goal_not_reached = torch.tensor([1 - goal_reached]).type(torch.int8)
            done = torch.ones_like(done) - (1 - done) * goal_not_reached
            
        next_tensordict[self.out_keys[0]] = done.type(torch.bool)
        
        return next_tensordict
    
    def transform_done_spec(self, done_spec):
        done_spec[self.out_keys[0]] = BoundedTensorSpec(
            low=0,
            high=1,
            shape=(1,),
            device=done_spec.device,
            dtype=torch.bool
        )
        
        return done_spec
