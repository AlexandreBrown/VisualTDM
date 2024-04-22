from comet_ml.exceptions import InterruptedExperiment
import hydra
import logging
import torch
import os
from omegaconf import DictConfig
from torchrl.collectors.collectors import SyncDataCollector
from tensordict.nn import TensorDictModule
from agents.td3_agent import Td3Agent
from envs.env_factory import create_env
from torchrl.modules import AdditiveGaussianWrapper
from torchrl.envs.transforms import RenameTransform
from torchrl.envs.transforms import DoubleToFloat
from torchrl.envs.transforms import CatTensors
from replay_buffers.factory import create_replay_buffer
from experiments.factory import create_experiment
from trainers.td3_trainer import Td3Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@hydra.main(version_base=None, config_path="configs/", config_name="td3_training")
def main(cfg: DictConfig):
    is_cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA AVAILABLE: {is_cuda_available}")
    
    models_device = torch.device(cfg['models']['device'])
    
    COMET_ML_API_KEY = os.getenv("COMET_ML_API_KEY")
    COMET_ML_PROJECT_NAME = os.getenv("COMET_ML_PROJECT_NAME")
    COMET_ML_WORKSPACE = os.getenv("COMET_ML_WORKSPACE")
    
    experiment = create_experiment(api_key=COMET_ML_API_KEY, project_name=COMET_ML_PROJECT_NAME, workspasce=COMET_ML_WORKSPACE)
    
    experiment.log_parameters(cfg)
    experiment.log_code(folder='src')
    experiment.log_other("cuda_available", is_cuda_available)

    train_env = create_env(cfg)
    train_env.append_transform(RenameTransform(in_keys=['observation'], out_keys=['state'], create_copy=False))
    train_env.append_transform(DoubleToFloat(in_keys=['desired_goal'], out_keys=['desired_goal']))
    train_env.append_transform(DoubleToFloat(in_keys=['achieved_goal'], out_keys=['achieved_goal']))
    train_env.append_transform(DoubleToFloat(in_keys=['state'], out_keys=['state']))
    train_env.append_transform(CatTensors(in_keys=list(cfg['models']['actor']['in_keys']), out_key="actor_inputs", del_keys=False))
    
    eval_env = create_env(cfg)
    eval_env.append_transform(RenameTransform(in_keys=['observation'], out_keys=['state'], create_copy=False))
    eval_env.append_transform(DoubleToFloat(in_keys=['desired_goal'], out_keys=['desired_goal']))
    eval_env.append_transform(DoubleToFloat(in_keys=['achieved_goal'], out_keys=['achieved_goal']))
    eval_env.append_transform(DoubleToFloat(in_keys=['state'], out_keys=['state']))
    eval_env.append_transform(CatTensors(in_keys=list(cfg['models']['actor']['in_keys']), out_key="actor_inputs", del_keys=False))
    
    actions_dim = train_env.action_spec.shape[0]
    action_space_low = train_env.action_spec.space.low
    action_space_high = train_env.action_spec.space.high
    action_scale = (action_space_high - action_space_low) / 2
    action_bias = (action_space_low + action_space_high) / 2
    actor_params = cfg['models']['actor']
    critic_params = cfg['models']['critic']
    state_dim = train_env.observation_spec['state'].shape[0]
    goal_dim = train_env.observation_spec['desired_goal'].shape[0]
    
    agent = Td3Agent(actor_model_type=actor_params['model_type'],
                         actor_hidden_layers_out_features=actor_params['hidden_layers_out_features'],
                         actor_hidden_activation_function_name=actor_params['hidden_activation_function_name'],
                         actor_output_activation_function_name=actor_params['output_activation_function_name'],
                         actor_learning_rate=cfg['train']['actor_learning_rate'],
                         critic_model_type=critic_params['model_type'],
                         critic_hidden_layers_out_features=critic_params['hidden_layers_out_features'],
                         critic_use_batch_norm=critic_params['use_batch_norm'],
                         critic_hidden_activation_function_name=critic_params['hidden_activation_function_name'],
                         critic_output_activation_function_name=critic_params['output_activation_function_name'],
                         critic_learning_rate=cfg['train']['critic_learning_rate'],
                         critic_gamma=cfg['train']['gamma'],
                         actions_dim=actions_dim,
                         action_scale=action_scale,
                         action_bias=action_bias,
                         goal_dim=goal_dim,
                         device=models_device,
                         polyak_avg=cfg['train']['polyak_avg'],
                         target_update_freq=cfg['train']['target_update_freq'],
                         target_policy_action_noise_clip=cfg['train']['target_policy_action_noise_clip'],
                         target_policy_action_noise_std=cfg['train']['target_policy_action_noise_std'],
                         state_dim=state_dim,
                         actor_in_keys=list(actor_params['in_keys']),
                         critic_in_keys=list(critic_params['in_keys']),
                         action_space_low=action_space_low,
                         action_space_high=action_space_high,
                         grad_norm_clipping=cfg['train']['grad_norm_clipping'])
 
    policy = TensorDictModule(agent.actor, in_keys="actor_inputs", out_keys=["action"])
    
    policy = AdditiveGaussianWrapper(policy=policy, 
                                                 sigma_init=cfg['train']['noise_sigma_init'],
                                                 sigma_end=cfg['train']['noise_sigma_end'],
                                                 annealing_num_steps=cfg['train']['noise_annealing_steps'],
                                                 mean=cfg['train']['noise_mean'],
                                                 std=cfg['train']['noise_std'],
                                                 action_key=policy.out_keys[0],
                                                 spec=train_env.action_spec,
                                                 safe=True)
    
    train_collector = SyncDataCollector(
        create_env_fn=train_env,
        policy=policy,
        total_frames=cfg['env']['total_frames'],
        init_random_frames=cfg['env']['init_random_frames'],
        max_frames_per_traj=cfg['env']['max_frames_per_traj'],
        frames_per_batch=cfg['env']['frames_per_batch'],
        reset_at_each_iter=cfg['env']['reset_at_each_iter'],
        device=torch.device(cfg['env']['collector_device']),
        storing_device=torch.device(cfg['env']['storing_device'])
    )
    
    replay_buffer = create_replay_buffer(cfg)
    
    trainer = Td3Trainer(experiment, train_collector, replay_buffer, cfg, agent, policy, logger, train_env, eval_env)
  
    try:
        trainer.train()
    except InterruptedExperiment as exc:
        experiment.log_other("status", str(exc))
        logger.info("Experiment interrupted!")
    
    logger.info("Training done!")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
