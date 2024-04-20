from comet_ml import Experiment
from loggers.simple_logger import SimpleLogger


class CometMlLogger:
    def __init__(self, experiment: Experiment, base_logger: SimpleLogger):
        self.experiment = experiment
        self.base_logger = base_logger
        self.step = 1
        self.episode = 1
        self.should_log = False
    
    def log_step_metric(self, key: str, value: float):
        if not self.should_log:
            return
        self.experiment.log_metric(name=f"{self.base_logger.step_log_prefix}{key}", value=value, step=self.step)
    
    def accumulate_step_metrics(self, metrics: dict):
        if not self.should_log:
            return
        self.base_logger.accumulate_step_metrics(metrics)
    
    def accumulate_step_metric(self, key: str, value: float):
        if not self.should_log:
            return
        self.base_logger.accumulate_step_metric(key, value)
    
    def compute_step_metrics(self) -> dict:
        if not self.should_log:
            return
        step_metrics = self.base_logger.compute_step_metrics()
        self.experiment.log_metrics(step_metrics, step=self.step)
        self.step += 1
        return step_metrics
    
    def accumulate_episode_metrics(self, metrics: dict):
        if not self.should_log:
            return
        self.base_logger.accumulate_episode_metrics(metrics)
    
    def accumulate_episode_metric(self, key: str, value: float):
        if not self.should_log:
            return
        self.base_logger.accumulate_episode_metric(key, value)
    
    def compute_episode_metrics(self) -> dict:
        if not self.should_log:
            return
        episode_metrics = self.base_logger.compute_episode_metrics()
        self.experiment.log_metrics(episode_metrics, epoch=self.episode)
        self.episode += 1
        return episode_metrics
