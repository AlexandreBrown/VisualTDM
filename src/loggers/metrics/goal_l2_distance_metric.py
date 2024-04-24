import torch
from tensordict import TensorDict
from loggers.metrics.metric import DefaultMetric


class GoalL2DistanceMetric(DefaultMetric):
    def __init__(self, achieved_goal_key: str, goal_key: str, name="goal_distance"):
        super().__init__(name=name)
        self.achieved_goal_key = achieved_goal_key
        self.goal_key = goal_key
    
    def compute(self, data: TensorDict) -> float:
        achieved_goal = data['next'][self.achieved_goal_key]
        goal = data[self.goal_key]
        distance = torch.linalg.vector_norm(goal - achieved_goal, ord=2, dim=-1)
        return distance.item()
