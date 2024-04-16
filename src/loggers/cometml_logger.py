from comet_ml import Experiment
from loggers.simple_logger import SimpleLogger


class CometMlLogger:
    def __init__(self, experiment: Experiment, base_logger: SimpleLogger):
        self.experiment = experiment
        self.base_logger = base_logger
        self.steps = 1
        self.episodes = 1
    
    def accumulate_step_metrics(self, metrics: dict):
        self.base_logger.accumulate_step_metrics(metrics)
    
    def accumulate_step_metric(self, key: str, value: float):
        self.base_logger.accumulate_step_metric(key, value)
    
    def compute_step_metrics(self) -> dict:
        step_metrics = self.base_logger.compute_step_metrics()
        self.experiment.log_metrics(step_metrics, step=self.steps)
        self.steps += 1
        return step_metrics
    
    def accumulate_episode_metrics(self, metrics: dict):
        self.base_logger.accumulate_episode_metrics(metrics)
    
    def accumulate_episode_metric(self, key: str, value: float):
        self.base_logger.accumulate_episode_metric(key, value)
    
    def compute_episode_metrics(self) -> dict:
        episode_metrics = self.base_logger.compute_episode_metrics()
        self.experiment.log_metrics(episode_metrics, epoch=self.episodes)
        self.episodes += 1
        return episode_metrics
