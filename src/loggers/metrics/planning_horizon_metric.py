from tensordict import TensorDict
from loggers.metrics.metric import DefaultMetric


class PlanningHorizonMetric(DefaultMetric):
    def __init__(self):
        super().__init__(name="planning_horizon")
    
    def compute(self, data: TensorDict) -> float:
        return float(data['planning_horizon'].item())
