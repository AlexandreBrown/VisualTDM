from tensordict import TensorDict
from loggers.metrics.goal_l2_distance_metric import GoalL2DistanceMetric
from loggers.metrics.metric import DefaultMetric


class GoalReachedMetric(DefaultMetric):
    def __init__(self, goal_distance_metric: GoalL2DistanceMetric):
        super().__init__(name="goal_reached")
        self.goal_distance_metric = goal_distance_metric
    
    def compute(self, data: TensorDict) -> float:
        distance = self.goal_distance_metric.compute(data)
        
        goal_reached = float(distance <= 0.05)
        
        return goal_reached
