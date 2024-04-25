from omegaconf import DictConfig
from loggers.metrics.episode_length_metric import EpisodeLengthMetric
from loggers.metrics.episode_return_metric import EpisodeReturnMetric
from loggers.metrics.goal_l2_distance_metric import GoalL2DistanceMetric
from loggers.metrics.goal_reached_metric import GoalReachedMetric
from loggers.metrics.metric import DefaultMetric
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
            goal_distance_metric = GoalL2DistanceMetric(achieved_goal_key="pixels_latent", goal_key="goal_latent")
            metric = GoalReachedMetric(cfg, goal_distance_metric, name="goal_latent_reached")
        elif metric_name == "goal_latent_distance":
            metric = GoalL2DistanceMetric(achieved_goal_key="pixels_latent", goal_key="goal_latent", name="goal_latent_distance")
        elif metric_name == "goal_reached":
            goal_distance_metric = GoalL2DistanceMetric(achieved_goal_key=achieved_goal_key, goal_key=goal_key)
            metric = GoalReachedMetric(cfg, goal_distance_metric)
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
    
    i = 0
    for metric_name in metric_names:
        if "episode_return" in metric_name:
            metric = EpisodeReturnMetric(name=metric_name, reward_key=cfg['logging']['reward_keys'][i])
            i += 1
        elif "episode_length" in metric_name:
            metric = EpisodeLengthMetric()
        metrics.append(metric)
    
    return metrics
