import torch
from tensordict import TensorDict
from torchrl.envs.transforms.transforms import Transform
from torchrl.data import BoundedTensorSpec
from envs.max_planning_horizon_scheduler import TdmMaxPlanningHorizonScheduler


class AddStepPlanningHorizon(Transform):
    """
    This represents the planning horizon that the policy will receive during a rollout.
    This is different than the planning horizon used by the Q-function during training.
    In the paper, this represents the T-t that is passed to the actor/MPC to get the action when acting in the env.
    Alternatively, we can experiment with tdm_rollout_max_planning_horizon = "Another value than the max_frames_per_traj".
    """
    def __init__(self,  tdm_rollout_max_planning_horizon: int, tdm_max_planning_horizon_scheduler: TdmMaxPlanningHorizonScheduler):
        super().__init__(in_keys=[], out_keys=["planning_horizon"])
        self.tdm_rollout_max_planning_horizon = tdm_rollout_max_planning_horizon
        self.tdm_max_planning_horizon_scheduler = tdm_max_planning_horizon_scheduler
        self.planning_horizon = self.get_max_planning_horizon()
    
    def get_max_planning_horizon(self):
        if self.tdm_rollout_max_planning_horizon is None:
            return self.tdm_max_planning_horizon_scheduler.get_max_planning_horizon()
        
        return self.tdm_rollout_max_planning_horizon
    
    def _call(self, tensordict: TensorDict):
        tensordict[self.out_keys[0]] = torch.full(size=(1,), fill_value=self.planning_horizon, device=tensordict.device, dtype=torch.float32)
        self.planning_horizon -= 1.
        if self.planning_horizon < 0:
            self.planning_horizon = self.get_max_planning_horizon()
        return tensordict
    
    def _reset(
        self, 
        tensordict: TensorDict, 
        tensordict_reset: TensorDict
    ) -> TensorDict:
        self.planning_horizon = self.get_max_planning_horizon()
        return self._call(tensordict_reset)
    
    def transform_observation_spec(self, observation_spec):
        observation_spec[self.out_keys[0]] = BoundedTensorSpec(
            low=1,
            high=self.tdm_rollout_max_planning_horizon if self.tdm_rollout_max_planning_horizon is not None else self.tdm_max_planning_horizon_scheduler.final_max_planning_horizon,
            shape=(1,),
            device=observation_spec.device,
            dtype=torch.float32
        )
        
        return observation_spec
