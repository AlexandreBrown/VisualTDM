from comet_ml.exceptions import InterruptedExperiment
from comet_ml import API
import hydra
import logging
import torch
import os
from pathlib import Path
from omegaconf import DictConfig
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.envs.transforms import ExcludeTransform
from tensordict.nn import TensorDictModule
from agents.tdm_td3_agent import TdmTd3Agent
from models.vae.model import VAEModel
from envs.max_planning_horizon_scheduler import TdmMaxPlanningHorizonScheduler
from envs.tdm_env_factory import create_tdm_env
from torchrl.modules import AdditiveGaussianWrapper
from experiments.factory import create_experiment
from replay_buffers.factory import create_replay_buffer
from trainers.tdm_td3_trainer import TdmTd3Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@hydra.main(version_base=None, config_path="configs/", config_name="tdm_training")
def main(cfg: DictConfig):
    is_cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA AVAILABLE: {is_cuda_available}")
    
    models_device = torch.device(cfg['models']['device'])
    
    COMET_ML_API_KEY = os.getenv("COMET_ML_API_KEY")
    COMET_ML_PROJECT_NAME = os.getenv("COMET_ML_PROJECT_NAME")
    COMET_ML_WORKSPACE = os.getenv("COMET_ML_WORKSPACE")
    
    encoder_decoder_model = download_encoder_decoder_model(api_key=COMET_ML_API_KEY, workspace=COMET_ML_WORKSPACE, cfg=cfg, device=models_device)
    
    experiment = create_experiment(api_key=COMET_ML_API_KEY, project_name=COMET_ML_PROJECT_NAME, workspasce=COMET_ML_WORKSPACE)
    
    experiment.log_parameters(cfg)
    experiment.log_code(folder='src')
    experiment.log_other("cuda_available", is_cuda_available)

    tdm_max_planning_horizon_scheduler = TdmMaxPlanningHorizonScheduler(initial_max_planning_horizon=cfg['train']['tdm_max_planning_horizon'],
                                                                        traj_max_nb_steps=cfg['env']['max_frames_per_traj'],
                                                                        total_frames=cfg['env']['total_frames'],
                                                                        step_batch_size=cfg['env']['frames_per_batch'],
                                                                        n_cycle=cfg['train']['tdm_planning_horizon_annealing_cycles'],
                                                                        ratio=cfg['train']['tdm_planning_horizon_annealing_ratio'],
                                                                        enable=cfg['train']['tdm_planning_horizon_annealing'])

    train_env = create_tdm_env(cfg, encoder_decoder_model, tdm_max_planning_horizon_scheduler)
    eval_env = create_tdm_env(cfg, encoder_decoder_model, tdm_max_planning_horizon_scheduler)
    
    actions_dim = train_env.action_spec.shape[0]
    action_space_low = train_env.action_spec.space.low
    action_space_high = train_env.action_spec.space.high
    action_scale = (action_space_high - action_space_low) / 2
    action_bias = (action_space_low + action_space_high) / 2
    actor_params = cfg['models']['actor']
    critic_params = cfg['models']['critic']
    state_dim = train_env.observation_spec['observation'].shape[0]
    
    agent = TdmTd3Agent(actor_model_type=actor_params['model_type'],
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
                            critic_is_relative=critic_params['is_relative'],
                            obs_dim=cfg['env']['obs']['dim'],
                            actions_dim=actions_dim,
                            action_scale=action_scale,
                            action_bias=action_bias,
                            goal_latent_dim=cfg['env']['goal']['latent_dim'],
                            device=models_device,
                            polyak_avg=cfg['train']['polyak_avg'],
                            distance_type=cfg['train']['reward_distance_type'],
                            target_update_freq=cfg['train']['target_update_freq'],
                            target_policy_action_noise_clip=cfg['train']['target_policy_action_noise_clip'],
                            target_policy_action_noise_std=cfg['train']['target_policy_action_noise_std'],
                            state_dim=state_dim,
                            actor_in_keys=list(actor_params['in_keys']),
                            critic_in_keys=list(critic_params['in_keys']),
                            action_space_low=action_space_low,
                            action_space_high=action_space_high,
                            reward_dim=cfg['train']['reward_dim'])
 
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
        storing_device=torch.device(cfg['env']['storing_device']),
        postproc=ExcludeTransform("pixels_transformed", ("next", "pixels_transformed"), "goal_pixels_transformed", ("next", "goal_pixels_transformed"))
    )
    
    replay_buffer = create_replay_buffer(cfg)
    
    trainer = TdmTd3Trainer(experiment, train_collector, replay_buffer, tdm_max_planning_horizon_scheduler, cfg, agent, policy, logger, train_env, eval_env, encoder_decoder_model)
  
    try:
        trainer.train()
    except InterruptedExperiment as exc:
        experiment.log_other("status", str(exc))
        logger.info("Experiment interrupted!")
    
    logger.info("Training done!")
    train_env.close()
    eval_env.close()


def download_encoder_decoder_model(api_key: str, workspace: str, cfg: DictConfig, device: torch.device) -> TensorDictModule:
    encoder_params = cfg['models']['encoder_decoder']['encoder']
    decoder_params = cfg['models']['encoder_decoder']['decoder']
    vae_model = VAEModel(input_spatial_dim=cfg['env']['obs']['height'],
                         input_channels=cfg['env']['obs']['dim'], 
                         encoder_hidden_dims=encoder_params['hidden_dims'],
                         encoder_hidden_activation=encoder_params['hidden_activation'],
                         encoder_hidden_kernels=encoder_params['hidden_kernels'],
                         encoder_hidden_strides=encoder_params['hidden_strides'],
                         encoder_hidden_paddings=encoder_params['hidden_paddings'],
                         encoder_use_batch_norm=encoder_params['use_batch_norm'],
                         encoder_leaky_relu_neg_slope=encoder_params['leaky_relu_neg_slope'],
                         latent_dim=cfg['env']['goal']['latent_dim'],
                         decoder_hidden_dims=decoder_params['hidden_dims'],
                         decoder_hidden_activation=decoder_params['hidden_activation'],
                         decoder_hidden_kernels=decoder_params['hidden_kernels'],
                         decoder_hidden_strides=decoder_params['hidden_strides'],
                         decoder_hidden_paddings=decoder_params['hidden_paddings'],
                         decoder_output_kernel=decoder_params['output_kernel'],
                         decoder_output_stride=decoder_params['output_stride'],
                         decoder_output_padding=decoder_params['output_padding'],
                         decoder_use_batch_norm=decoder_params['use_batch_norm']).to(device)
    
    api = API(api_key=api_key)
    encoder_decoder_model = api.get_model(workspace=workspace, model_name=cfg['models']['encoder_decoder']['name'])
    hydra_output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    model_folder = Path(cfg['models']['encoder_decoder']['download_path'])
    output_folder = hydra_output_path / model_folder
    output_folder.mkdir(parents=True)
    encoder_decoder_model.download(version=cfg['models']['encoder_decoder']['version'], output_folder=output_folder)
    model_weights_path = output_folder / Path("model-data") / Path("comet-torch-model.pth")
    
    encoder_decoder_model = TensorDictModule(vae_model, in_keys=["image"], out_keys=["q_z", "p_x"])
    state_dict = torch.load(model_weights_path)
    del state_dict['__batch_size']
    del state_dict['__device']
    encoder_decoder_model.load_state_dict(state_dict)
    
    for param in encoder_decoder_model.parameters():
        param.requires_grad = False
    
    return encoder_decoder_model


if __name__ == "__main__":
    main()
