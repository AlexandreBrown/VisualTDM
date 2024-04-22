from omegaconf import DictConfig
from tensordict import TensorDict
from loggers.metrics.metric import DefaultMetric
from rewards.distance import compute_distance


class GoalLatentDistanceMetric(DefaultMetric):
    def __init__(self, cfg: DictConfig):
        super().__init__(name="goal_latent_distance")
        self.distance_type = cfg['train']['reward_distance_type']
        self.goal_reached_epsilon = cfg['env']['goal']['reached_epsilon']
    
    def compute(self, data: TensorDict) -> float:
        next_obs_latent = data['next']['pixels_latent']
        goal_latent = data['goal_latent']
        
        distance = compute_distance(distance_type=self.distance_type, state=next_obs_latent, goal=goal_latent).mean()
        
        return distance.item()
