from omegaconf import DictConfig
from tensordict import TensorDict
from loggers.metrics.metric import DefaultMetric
from tensor_utils import get_tensor

class QValueMetric(DefaultMetric):
    def __init__(self, critic, cfg: DictConfig):
        super().__init__(name="q_value")
        self.critic = critic
        self.critic_in_keys = list(cfg['models']['critic']['in_keys'])
    
    def compute(self, data: TensorDict) -> float:
        critic_input = get_tensor(data, self.critic_in_keys).unsqueeze(0)
        self.critic.eval()
        q_value = self.critic(critic_input)
        return q_value.mean()
