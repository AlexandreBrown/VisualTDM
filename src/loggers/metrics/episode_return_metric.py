from tensordict import TensorDict
from loggers.metrics.metric import DefaultMetric


class EpisodeReturnMetric(DefaultMetric):
    def __init__(self):
        super().__init__(name="episode_return")
    
    def compute(self, episode_data: TensorDict) -> float:
        return episode_data['next']['reward'].sum().item()
