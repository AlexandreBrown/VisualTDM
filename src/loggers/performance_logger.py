import hydra
from omegaconf import DictConfig
from pathlib import Path
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase
from torchrl.envs.utils import set_exploration_type
from torchrl.envs.utils import ExplorationType
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record import VideoRecorder
from loggers.cometml_logger import CometMlLogger


class PerformanceLogger:
    def __init__(self, base_logger: CometMlLogger, env: EnvBase, cfg: DictConfig, eval_policy: TensorDictModule, metrics: list):
        self.base_logger = base_logger
        self.env = env
        self.cfg = cfg
        self.eval_policy = eval_policy
        self.video_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / Path(self.cfg['logging']['video_dir'])
        self.video_dir.mkdir(parents=True, exist_ok=True)
        video_logger = CSVLogger(exp_name=cfg['experiment']['name'], log_dir=self.video_dir, video_format="mp4", video_fps=cfg['logging']['video_fps'])
        self.recorder = VideoRecorder(logger=video_logger, tag="step", skip=1)
        self.metrics = metrics
    
    def log_step(self, step: int):
        if step % self.cfg['logging']['step_freq'] != 0 or \
            (step+1) * self.cfg['env']['frames_per_batch'] <= self.cfg['env']['init_random_frames']:
            return

        with set_exploration_type(type=ExplorationType.MEAN):
            data = self.env.rollout(max_steps=self.cfg['logging']['metrics_frames'], policy=self.eval_policy, break_when_any_done=False)
            
            for step_data in data:
                for metric in self.metrics:
                    key = metric.name
                    value = metric.compute(step_data)
                    self.base_logger.accumulate_step_metric(key=key, value=value)
            
            self.base_logger.compute_step_metrics(step=step)
            self._log_video(data, step)
    
    def _log_video(self, data: TensorDict, step: int):
        if step % self.cfg['logging']['video_log_step_freq'] != 0:
            return
        
        for obs in data['pixels']:
            self.recorder._apply_transform(obs)
        self.recorder.dump()
        video_files = list((self.video_dir / Path(self.cfg['experiment']['name']) / Path('videos')).glob("*.mp4"))
        video_files.sort(reverse=True)
        video_file = video_files[0]
        self.base_logger.log_step_video(video_file, step)
