from omegaconf import DictConfig
import torch
from tensordict import TensorDict
from loggers.metrics.metric import DefaultMetric
from loggers.metrics.goal_latent_distance_metric import GoalLatentDistanceMetric


class GoalLatentReachedMetric(DefaultMetric):
    def __init__(self, cfg: DictConfig, goal_latent_distance_metric: GoalLatentDistanceMetric):
        super().__init__(name="goal_latent_reached")
        self.goal_latent_distance_metric = goal_latent_distance_metric
        self.goal_reached_epsilon = cfg['env']['goal']['reached_epsilon']
    
    def compute(self, data: TensorDict) -> float:
        distance = self.goal_latent_distance_metric.compute(data)
        
        goal_reached = float(distance <= self.goal_reached_epsilon)
        
        return goal_reached
