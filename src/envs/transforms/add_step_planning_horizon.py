import torch
from tensordict import TensorDict
from torchrl.envs.transforms.transforms import Transform
from torchrl.data import BoundedTensorSpec


class AddStepPlanningHorizon(Transform):
    """
    This represents the planning horizon that the policy will receive during a rollout.
    This is different than the planning horizon used by the Q-values during training.
    In the paper, this represents the T-t that is passed to the actor/MPC to get the action when acting in the env.
    """
    def __init__(self,  traj_max_nb_steps: int):
        super().__init__(in_keys=[], out_keys=["planning_horizon"])
        self.traj_max_nb_steps = traj_max_nb_steps
        self.planning_horizon = traj_max_nb_steps
    
    def _call(self, tensordict: TensorDict):
        tensordict[self.out_keys[0]] = torch.full(size=(1,), fill_value=self.planning_horizon, device=tensordict.device)
        self.planning_horizon -= 1.
        return tensordict
    
    def _reset(
        self, 
        tensordict: TensorDict, 
        tensordict_reset: TensorDict
    ) -> TensorDict:
        self.planning_horizon = self.traj_max_nb_steps
        return self._call(tensordict_reset)
    
    def transform_observation_spec(self, observation_spec):
        observation_spec[self.out_keys[0]] = BoundedTensorSpec(
            low=1,
            high=self.traj_max_nb_steps,
            shape=(1,),
            device=observation_spec.device,
        )
        
        return observation_spec
