import torch
import math
from annealing.cyclical_linear import frange_cycle_linear


class TdmMaxPlanningHorizonScheduler:
    def __init__(self, 
                 initial_max_planning_horizon: int, 
                 traj_max_nb_steps: int,
                 total_frames: int,
                 step_batch_size: int,
                 n_cycle: int,
                 ratio: float,
                 enable: bool):
        self.initial_max_planning_horizon = initial_max_planning_horizon
        self.final_max_planning_horizon = traj_max_nb_steps - 1
        n_iter = int(math.ceil(total_frames / step_batch_size))
        if enable:
            self.planning_horizon_schedule = frange_cycle_linear(n_iter=n_iter, start=0.0, stop=1.0,  n_cycle=n_cycle, ratio=ratio)
        else:
            self.planning_horizon_schedule = torch.ones(size=(n_iter,))
        self.t = 0
        self.enable = enable
    
    def get_max_planning_horizon(self):
        candidate_max_planning_horzion = int(self.planning_horizon_schedule[self.t] * self.final_max_planning_horizon)
        
        if self.t == 0 or candidate_max_planning_horzion < self.initial_max_planning_horizon:
            return self.initial_max_planning_horizon
        
        return candidate_max_planning_horzion
    
    def step(self, trained: bool):
        if not self.enable or not trained:
            return
        if self.t < self.planning_horizon_schedule.shape[0]:
            self.t += 1
