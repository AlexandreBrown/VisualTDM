from tensordict import TensorDict
from loggers.metrics.metric import DefaultMetric


class EpisodeReturnMetric(DefaultMetric):
    def __init__(self, name: str="episode_return", reward_key: str="reward"):
        super().__init__(name=name)
        self.reward_key = reward_key
    
    def compute(self, episode_data: TensorDict) -> float:
        return episode_data['next'][self.reward_key].sum().item()
