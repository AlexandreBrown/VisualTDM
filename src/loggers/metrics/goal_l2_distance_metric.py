from tensordict import TensorDict
from loggers.metrics.metric import DefaultMetric
from rewards.distance import compute_distance


class GoalL2DistanceMetric(DefaultMetric):
    def __init__(self, achieved_goal_key: str, goal_key: str):
        super().__init__(name="goal_distance")
        self.distance_type = 'l2'
        self.achieved_goal_key = achieved_goal_key
        self.goal_key = goal_key
    
    def compute(self, data: TensorDict) -> float:
        next_state = data['next'][self.achieved_goal_key]
        goal = data[self.goal_key]
        
        distance = compute_distance(distance_type=self.distance_type, state=next_state, goal=goal).mean()
        
        return distance.item()
