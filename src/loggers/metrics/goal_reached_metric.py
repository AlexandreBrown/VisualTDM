from omegaconf import DictConfig
from tensordict import TensorDict
from loggers.metrics.goal_l2_distance_metric import GoalL2DistanceMetric
from loggers.metrics.metric import DefaultMetric


class GoalReachedMetric(DefaultMetric):
    def __init__(self, cfg: DictConfig, goal_distance_metric: GoalL2DistanceMetric, name: str = "goal_reached"):
        super().__init__(name=name)
        self.goal_distance_metric = goal_distance_metric
        self.goal_reached_epsilon = cfg['env']['goal']['reached_epsilon']
    
    def compute(self, data: TensorDict) -> float:
        distance = self.goal_distance_metric.compute(data)
        
        goal_reached = float(distance <= self.goal_reached_epsilon)
        
        return goal_reached
