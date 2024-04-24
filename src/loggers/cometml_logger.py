from comet_ml import Experiment
from loggers.simple_logger import SimpleLogger


class CometMlLogger:
    def __init__(self, experiment: Experiment, base_logger: SimpleLogger):
        self.experiment = experiment
        self.base_logger = base_logger
    
    def log_step_metric(self, key: str, value: float, step: int):
        self.experiment.log_metric(name=f"{self.base_logger.step_log_prefix}{key}", value=value, step=step+1)
    
    def accumulate_step_metrics(self, metrics: dict):
        self.base_logger.accumulate_step_metrics(metrics)
    
    def accumulate_step_metric(self, key: str, value: float):
        self.base_logger.accumulate_step_metric(key, value)
    
    def compute_step_metrics(self, step: int) -> dict:
        step_metrics = self.base_logger.compute_step_metrics()
        self.experiment.log_metrics(step_metrics, step=step+1)
        return step_metrics
    
    def log_step_video(self, video_file, step: int):
        self.experiment.log_video(file=video_file, name=f"{self.base_logger.stage_prefix}rollouts_step{step+1}", step=step+1)
    
    def log_step_image(self, image_data, step: int, rollout_id: int, name: str):
        self.experiment.log_image(image_data=image_data, name=f"{self.base_logger.stage_prefix}rollout{rollout_id}_{name}_step{step+1}", step=step+1)
