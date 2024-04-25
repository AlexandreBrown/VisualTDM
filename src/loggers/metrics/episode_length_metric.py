from tensordict import TensorDict
from loggers.metrics.metric import DefaultMetric


class EpisodeLengthMetric(DefaultMetric):
    def __init__(self):
        super().__init__(name="episode_length")
    
    def compute(self, episode_data: TensorDict) -> float:
        return float(episode_data.shape[0])
