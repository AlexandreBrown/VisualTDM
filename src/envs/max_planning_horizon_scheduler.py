from annealing.cyclical_linear import frange_cycle_linear


class MaxPlanningHorizonScheduler:
    def __init__(self, 
                 initial_max_planning_horizon: int, 
                 steps_per_traj: int,
                 n_cycle: int,
                 ratio: float):
        self.initial_max_planning_horizon = initial_max_planning_horizon
        self.final_max_planning_horizon = steps_per_traj - 1
        self.planning_horizon_schedule = frange_cycle_linear(n_iter=steps_per_traj, start=0.0, stop=1.0,  n_cycle=n_cycle, ratio=ratio)
        self.t = 0
    
    def get_max_planning_horizon(self):
        candidate_max_planning_horzion = int(self.planning_horizon_schedule[self.t] * self.final_max_planning_horizon)
        
        if self.t == 0 or candidate_max_planning_horzion < self.initial_max_planning_horizon:
            return self.initial_max_planning_horizon
        
        return candidate_max_planning_horzion
    
    def step(self):
        if self.t < self.planning_horizon_schedule.shape[0]:
            self.t += 1
