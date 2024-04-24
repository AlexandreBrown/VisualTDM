import logging
import torch
from comet_ml import Experiment
from omegaconf import DictConfig
from torchrl.collectors.collectors import DataCollectorBase
from torchrl.data.replay_buffers import ReplayBuffer
from tensordict.nn import TensorDictModule
from agents.tdm_td3_agent import TdmTd3Agent
from models.vae.model import VAEModel
from envs.max_planning_horizon_scheduler import TdmMaxPlanningHorizonScheduler
from tensordict import TensorDict
from loggers.simple_logger import SimpleLogger
from loggers.cometml_logger import CometMlLogger
from torchrl.envs import EnvBase
from replay_buffers.utils import get_step_data_of_interest
from loggers.performance_logger import PerformanceLogger
from loggers.metrics.factory import create_step_metrics
from loggers.metrics.factory import create_episode_metrics
from envs.transforms.add_tdm_done import AddTdmDone
from loggers.metrics.goal_reached_metric import GoalReachedMetric
from loggers.metrics.goal_l2_distance_metric import GoalL2DistanceMetric
from envs.transforms.add_goal_vector_distance_reward import AddGoalVectorDistanceReward


class TdmTd3Trainer:
    def __init__(self, experiment: Experiment, train_collector: DataCollectorBase, replay_buffer: ReplayBuffer, tdm_max_planning_horizon_scheduler: TdmMaxPlanningHorizonScheduler, cfg: DictConfig, agent: TdmTd3Agent, policy: TensorDictModule, logger: logging.Logger, train_env: EnvBase, eval_env: EnvBase, encoder_decoder_model: VAEModel):
        self.experiment = experiment
        self.train_collector = train_collector
        self.replay_buffer = replay_buffer
        self.tdm_max_planning_horizon_scheduler = tdm_max_planning_horizon_scheduler
        self.cfg = cfg
        self.agent = agent
        self.policy = policy
        self.logger = logger
        self.train_env = train_env
        self.eval_env = eval_env
        self.encoder_decoder_model = encoder_decoder_model
        self.train_stage_prefix = "train_"
        self.eval_stage_prefix = "eval_"
        self.tdm_done_transform = AddTdmDone(terminate_when_goal_reached=cfg['train']['tdm_terminate_when_goal_reached'], goal_latent_reached_metric=GoalReachedMetric(cfg, GoalL2DistanceMetric(achieved_goal_key="pixels_latent", goal_key="goal_latent")))
        self.reward_transform = AddGoalVectorDistanceReward(cfg['train']['reward_distance_type'], reward_dim=cfg['env']['goal']['latent_dim'])
        
    def train(self):
        self.logger.info("Starting training...")
        
        eval_metrics = create_step_metrics(self.cfg, critic=self.agent.critic)
        eval_episode_metrics = create_episode_metrics(self.cfg)
        eval_logger = PerformanceLogger(base_logger=CometMlLogger(experiment=self.experiment, base_logger=SimpleLogger(stage_prefix=self.eval_stage_prefix)), env=self.eval_env, cfg=self.cfg, eval_policy=self.policy, step_metrics=eval_metrics, episode_metrics=eval_episode_metrics)
        
        for step, data in enumerate(self.train_collector):
            traj_ids = data['collector']['traj_ids']
            step_data = get_step_data_of_interest(data=data, cfg=self.cfg)
            step_data['traj'] = traj_ids
            self.replay_buffer.extend(step_data)
            
            running_env_steps = (step+1) * data.shape[0] - 1
            
            eval_logger.log_step(running_env_steps)
            
            self.do_train_updates(running_env_steps)

            self.policy.step(data.shape[0])
            self.tdm_max_planning_horizon_scheduler.step(data.shape[0])

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
        can_sample_train_batch = len(self.replay_buffer) >= train_batch_size
        
        return is_random_exploration_over and can_sample_train_batch

    def get_is_random_exploration_over(self) -> bool:
        return len(self.replay_buffer) >= self.cfg['env']['init_random_frames']

    def do_train_update(self, train_batch_size: int) -> dict:
        train_data_sample = self.replay_buffer.sample(train_batch_size)
        
        train_data_sample = self.relabel_train_data(train_data_sample)
        
        train_update_metrics = self.agent.train(train_data_sample)
        
        return train_update_metrics

    def relabel_train_data(self, train_data_sample: TensorDict) -> TensorDict:
        train_data_sample_relabeled = train_data_sample.clone(recurse=True)

        train_data_sample_relabeled = self.relabel_planning_horizon(train_data_sample_relabeled)
        train_data_sample_relabeled = self.relabel_goal(train_data_sample_relabeled)
        
        return train_data_sample_relabeled

    def relabel_planning_horizon(self, train_data_sample_relabeled: TensorDict) -> TensorDict:
        batch_size = train_data_sample_relabeled.batch_size[0]
        
        new_planning_horizon = torch.randint(low=0, high=self.tdm_max_planning_horizon_scheduler.get_max_planning_horizon() + 1, size=(batch_size, 1))
        
        train_data_sample_relabeled['planning_horizon'] = new_planning_horizon
        train_data_sample_relabeled['next']['planning_horizon'] = torch.max(new_planning_horizon - 1, torch.zeros_like(new_planning_horizon))
        
        return train_data_sample_relabeled

    def relabel_goal(self, train_data_sample_relabeled: TensorDict) -> TensorDict:
        relabel_index = -1
        train_data_sample_goal_relabeled = train_data_sample_relabeled.clone(recurse=True)
        for traj_id in torch.unique_consecutive(train_data_sample_relabeled['traj']):
            traj_mask = train_data_sample_relabeled['traj'] == traj_id
            traj_data = train_data_sample_relabeled[traj_mask]
            traj_length = traj_data.shape[0]
            
            for traj_step in range(traj_length - 1):
                
                relabel_index += 1
                
                traj_low_index = traj_step + 1
                traj_high_index = traj_length - 1
                
                if traj_low_index == traj_high_index:
                    train_data_sample_goal_relabeled['goal_latent'][[relabel_index]] = traj_data['pixels_latent'][traj_low_index]
                else:
                    random_future_index = torch.randint(low=traj_low_index, high=traj_high_index, size=(1,))
                    train_data_sample_goal_relabeled['goal_latent'][relabel_index] = traj_data['pixels_latent'][random_future_index]

            train_data_sample_goal_relabeled['next'][traj_mask] = self.reward_transform._step(train_data_sample_goal_relabeled[traj_mask], train_data_sample_goal_relabeled['next'][traj_mask])
            
            done_steps_tensordicts = []
            for t in range(traj_length):
                current_step = train_data_sample_goal_relabeled[traj_mask][t]
                next_step = train_data_sample_goal_relabeled['next'][traj_mask][t]
                
                next_step_with_done = self.tdm_done_transform._step(current_step, next_step)
                done_steps_tensordicts.append(next_step_with_done)
                
            train_data_sample_goal_relabeled['next'][traj_mask] = torch.stack(done_steps_tensordicts)
            
            relabel_index += 1
        
        return train_data_sample_goal_relabeled
