from omegaconf import DictConfig
from loggers.metrics.episode_return_metric import EpisodeReturnMetric
from loggers.metrics.goal_l2_distance_metric import GoalL2DistanceMetric
from loggers.metrics.goal_latent_distance_metric import GoalLatentDistanceMetric
from loggers.metrics.goal_reached_metric import GoalReachedMetric
from loggers.metrics.metric import DefaultMetric
from loggers.metrics.goal_latent_reached_metric import GoalLatentReachedMetric
from loggers.metrics.planning_horizon_metric import PlanningHorizonMetric
from loggers.metrics.q_value_metric import QValueMetric


def create_step_metrics(cfg: DictConfig,
                   achieved_goal_key: str = "achieved_goal", 
                   goal_key: str = "desired_goal",
                   critic = None) -> list:
    metric_names = list(cfg['logging']['step_metrics'])
    metrics = []
    
    for metric_name in metric_names:
        if metric_name == "goal_latent_reached":
            goal_latent_distance_metric = GoalLatentDistanceMetric(cfg)
            metric = GoalLatentReachedMetric(cfg, goal_latent_distance_metric)
        elif metric_name == "goal_latent_distance":
            metric = GoalLatentDistanceMetric(cfg)
        elif metric_name == "goal_reached":
            goal_distance_metric = GoalL2DistanceMetric(achieved_goal_key=achieved_goal_key, goal_key=goal_key)
            metric = GoalReachedMetric(goal_distance_metric)
        elif metric_name == "goal_distance":
            metric = GoalL2DistanceMetric(achieved_goal_key=achieved_goal_key, goal_key=goal_key)
        elif metric_name == "planning_horizon":
            metric = PlanningHorizonMetric()
        elif metric_name == "q_value":
            metric = QValueMetric(critic, cfg)
        else:
            metric = DefaultMetric(name=metric_name)
        metrics.append(metric)
    
    return metrics


def create_episode_metrics(cfg: DictConfig) -> list:
    metric_names = list(cfg['logging']['episode_metrics'])
    metrics = []
    
    for metric_name in metric_names:
        if metric_name == "episode_return":
            metric = EpisodeReturnMetric()
        metrics.append(metric)
    
    return metrics
