import torch
import logging
from comet_ml import Experiment
from omegaconf import DictConfig
from torchrl.collectors.collectors import DataCollectorBase
from torchrl.data.replay_buffers import ReplayBuffer
from agents.td3_agent import Td3Agent
from torchrl.envs import EnvBase
from tensordict.nn import TensorDictModule
from replay_buffers.utils import get_step_data_of_interest
from loggers.performance_logger import PerformanceLogger
from loggers.metrics.factory import create_metrics
from loggers.simple_logger import SimpleLogger
from loggers.cometml_logger import CometMlLogger


class Td3Trainer:
    def __init__(self, experiment: Experiment, train_collector: DataCollectorBase, replay_buffer: ReplayBuffer, cfg: DictConfig, agent: Td3Agent, policy: TensorDictModule, logger: logging.Logger, train_env: EnvBase, eval_env: EnvBase):
        self.experiment = experiment
        self.train_collector = train_collector
        self.replay_buffer = replay_buffer
        self.cfg = cfg
        self.agent = agent
        self.policy = policy
        self.logger = logger
        self.train_env = train_env
        self.eval_env = eval_env
        self.train_stage_prefix = "train_"
        self.eval_stage_prefix = "eval_"
    
    def train(self):
        self.logger.info("Starting training...")
        
        train_metrics = create_metrics(self.cfg, critic=self.agent.critic)
        train_logger = PerformanceLogger(base_logger=CometMlLogger(experiment=self.experiment, base_logger=SimpleLogger(stage_prefix=self.train_stage_prefix)), env=self.train_env, cfg=self.cfg, eval_policy=self.policy, metrics=train_metrics)
        
        eval_metrics = create_metrics(self.cfg, critic=self.agent.critic)
        eval_logger = PerformanceLogger(base_logger=CometMlLogger(experiment=self.experiment, base_logger=SimpleLogger(stage_prefix=self.eval_stage_prefix)), env=self.eval_env, cfg=self.cfg, eval_policy=self.policy, metrics=eval_metrics)
        
        for step, data in enumerate(self.train_collector):
            traj_ids = data['collector']['traj_ids']
            step_data = get_step_data_of_interest(data=data, cfg=self.cfg)
            step_data['traj'] = traj_ids
            self.replay_buffer.extend(step_data)

            train_logger.log_step(step)
            eval_logger.log_step(step)
            
            self.do_train_updates(step)

            self.policy.step(self.cfg['env']['frames_per_batch'])

    def do_train_updates(self, step: int):
        train_batch_size = self.cfg['train']['batch_size']
        
        if not self.can_train(train_batch_size, step):
            return

        train_updates_logger = CometMlLogger(self.experiment, SimpleLogger(stage_prefix=self.train_stage_prefix))
        
        for _ in range(self.cfg['train']['updates_per_step']):
            train_update_metrics = self.do_train_update(train_batch_size)
            train_updates_logger.accumulate_step_metrics(train_update_metrics)
        
        train_updates_logger.compute_step_metrics(step=step)

    def can_train(self, train_batch_size: int, step: int) -> bool:
        is_random_exploration_over = self.get_is_random_exploration_over()
        is_learning_step = step % self.cfg['train']['learning_frequency'] == 0
        can_sample_train_batch = len(self.replay_buffer) >= train_batch_size
        
        return is_random_exploration_over and is_learning_step and can_sample_train_batch

    def get_is_random_exploration_over(self) -> bool:
        return len(self.replay_buffer) >= self.cfg['env']['init_random_frames']

    def do_train_update(self, train_batch_size: int):
        train_data_sample = self.replay_buffer.sample(train_batch_size)
        
        
        if self.cfg['train']['relabel_goal'] and torch.rand(1) <= self.cfg['train']['relabel_p']:
            train_data_sample_relabeled = train_data_sample.clone(recurse=True)
            
            min_goal = torch.min(train_data_sample['next']['achieved_goal'], dim=0).values
            min_goal = min_goal * 0.95
            max_goal = torch.max(train_data_sample['next']['achieved_goal'], dim=0).values
            max_goal = max_goal * 1.05
            goal_range = max_goal - min_goal
            new_goal = min_goal + torch.rand(train_data_sample['desired_goal'].shape[0], train_data_sample['desired_goal'].shape[1]) * goal_range
            
            train_data_sample_relabeled['desired_goal'] = new_goal
            distance = torch.linalg.vector_norm(new_goal - train_data_sample_relabeled['achieved_goal'], ord=2, dim=1)
            train_data_sample_relabeled['next']['reward'] = -distance
            train_data_sample_relabeled['next']['done'] = distance <= self.cfg['env']['goal']['reached_epsilon']
        else:
            train_data_sample_relabeled = train_data_sample
        
        train_update_metrics = self.agent.train(train_data_sample_relabeled)
        
        return train_update_metrics
