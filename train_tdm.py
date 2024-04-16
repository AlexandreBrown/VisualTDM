from comet_ml import Experiment
from comet_ml.exceptions import InterruptedExperiment
from comet_ml import API
import hydra
import logging
import torch
import os
from pathlib import Path
from omegaconf import DictConfig
from envs.env_factory import create_env
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.collectors.collectors import DataCollectorBase
from torchrl.data.replay_buffers import LazyMemmapStorage
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.transforms import Compose
from torchrl.envs.transforms import ToTensorImage
from torchrl.envs.transforms import Resize
from torchrl.envs.transforms import ExcludeTransform
from torchrl.envs.transforms import ObservationNorm
from tensordict.nn import TensorDictModule
from envs.transforms.add_planning_horizon import AddPlanningHorizon
from envs.transforms.add_goal_latent_representation import AddGoalLatentRepresentation
from agents.tdm_agent import TdmAgent
from torchrl.modules.tensordict_module import AdditiveGaussianWrapper
from models.vae.model import VAEModel
from envs.transforms.compute_latent_goal_distance_vector_reward import ComputeLatentGoalDistanceVectorReward
from envs.max_planning_horizon_scheduler import MaxPlanningHorizonScheduler
from torchrl.envs.utils import set_exploration_type
from torchrl.envs.utils import ExplorationType
from tensordict import TensorDict
from loggers.simple_logger import SimpleLogger
from loggers.cometml_logger import CometMlLogger
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record import VideoRecorder

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
    
    env_device = torch.device(cfg['env']['device'])
    
    train_env = create_env(env_name=cfg['env']['name'],
                           seed=cfg['experiment']['seed'],
                           device=env_device,
                           normalize_obs=cfg['env']['obs']['normalize'],
                           standardization_stats_init_iter=cfg['env']['obs']['standardization_stats_init_iter'],
                           standardize_obs=cfg['env']['obs']['standardize'],
                           raw_height=cfg['env']['obs']['raw_height'],
                           raw_width=cfg['env']['obs']['raw_width'],
                           resize_dim=(cfg['env']['obs']['width'], cfg['env']['obs']['height']))
    
    max_planning_horizon_scheduler = MaxPlanningHorizonScheduler(initial_max_planning_horizon=cfg['train']['initial_max_planning_horizon'],
                                                          steps_per_traj=cfg['train']['max_frames_per_traj'],
                                                          n_cycle=cfg['train']['planning_horizon_annealing_cycles'],
                                                          ratio=cfg['train']['planning_horizon_annealing_ratio'],
                                                          enable=cfg['train']['planning_horizon_annealing'])
    
    train_env.append_transform(AddPlanningHorizon(max_planning_horizon_scheduler=max_planning_horizon_scheduler))
    train_env.append_transform(AddGoalLatentRepresentation(encoder_decoder_model=encoder_decoder_model,
                                                           latent_dim=cfg['env']['goal']['latent_dim']))
    train_env.append_transform(ComputeLatentGoalDistanceVectorReward(norm_type=cfg['train']['reward_norm_type'],
                                                                     encoder=encoder_decoder_model,
                                                                     latent_dim=cfg['env']['goal']['latent_dim']))
    
    actions_dim = train_env.action_spec.shape[0]
    action_space_low = train_env.action_spec.space.low
    action_space_high = train_env.action_spec.space.high
    action_scale = (action_space_high - action_space_low) / 2
    action_bias = (action_space_low + action_space_high) / 2
    actor_params = cfg['models']['actor']
    critic_params = cfg['models']['critic']
    tdm_agent = TdmAgent(actor_model_type=actor_params['model_type'],
                         actor_hidden_layers_out_features=actor_params['hidden_layers_out_features'],
                         actor_hidden_activation_function_name=actor_params['hidden_activation_function_name'],
                         actor_output_activation_function_name=actor_params['output_activation_function_name'],
                         actor_learning_rate=cfg['train']['actor_learning_rate'],
                         critic_model_type=critic_params['model_type'],
                         critic_hidden_layers_out_features=critic_params['hidden_layers_out_features'],
                         critic_hidden_activation_function_name=critic_params['hidden_activation_function_name'],
                         critic_output_activation_function_name=critic_params['output_activation_function_name'],
                         critic_learning_rate=cfg['train']['critic_learning_rate'],
                         obs_dim=cfg['env']['obs']['dim'],
                         actions_dim=actions_dim,
                         action_scale=action_scale,
                         action_bias=action_bias,
                         goal_latent_dim=cfg['env']['goal']['latent_dim'],
                         device=models_device,
                         polyak_avg=cfg['train']['polyak_avg'],
                         encoder=encoder_decoder_model,
                         norm_type=cfg['train']['reward_norm_type'],
                         target_update_freq=cfg['train']['target_update_freq'])
 
    policy = TensorDictModule(tdm_agent.actor, in_keys=["pixels_transformed", "goal_latent", "planning_horizon"], out_keys=["action"])
    
    exploration_policy = AdditiveGaussianWrapper(policy=policy, spec=train_env.action_spec, mean=cfg['train']['noise_mean'], std=cfg['train']['noise_std'], annealing_num_steps=cfg['train']['noise_annealing_frames'])
    
    train_collector = SyncDataCollector(
        create_env_fn=train_env,
        policy=exploration_policy,
        total_frames=-1,
        init_random_frames=cfg['train']['init_random_frames'],
        max_frames_per_traj=cfg['train']['max_frames_per_traj'],
        frames_per_batch=cfg['train']['frames_per_batch'],
        reset_at_each_iter=cfg['train']['reset_at_each_iter'],
        device=torch.device(cfg['train']['collector_device']),
        storing_device=torch.device(cfg['train']['storing_device']),
        postproc=ExcludeTransform("pixels_transformed", ("next", "pixels_transformed"), "goal_pixels_transformed", ("next", "goal_pixels_transformed"))
    )
    
    rb = create_replay_buffer(train_env, cfg, encoder_decoder_model, max_planning_horizon_scheduler)
  
    try:
        train(experiment, train_collector, rb, max_planning_horizon_scheduler, cfg, tdm_agent, exploration_policy, logger)          
    except InterruptedExperiment as exc:
        experiment.log_other("status", str(exc))
        logger.info("Experiment interrupted!")
    
    train_env.close()


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
    
    return encoder_decoder_model


def create_experiment(api_key: str, project_name: str, workspasce: str) -> Experiment:
    return Experiment(
        api_key=api_key,
        project_name=project_name,
        workspace=workspasce
    )


def create_replay_buffer(train_env, cfg: DictConfig, encoder_decoder_model: TensorDictModule, max_planning_horizon_scheduler: MaxPlanningHorizonScheduler) -> ReplayBuffer:
    rb_transforms = []
    if cfg['env']['obs']['normalize']:
        rb_transforms.append(ToTensorImage(in_keys=["pixels", ("next", "pixels")], out_keys=["pixels_transformed", ("next", "pixels_transformed")]))
    rb_transforms.append(Resize(in_keys=["pixels_transformed", ("next", "pixels_transformed")], w=cfg['env']['obs']['width'], h=cfg['env']['obs']['height']))
    
    obs_loc = 0.
    obs_scale = 1.
    
    if cfg['env']['obs']['standardize']:
        obs_loc = train_env.transform[-1].loc
        obs_scale = train_env.transform[-1].scale
        rb_transforms.append(ObservationNorm(loc=obs_loc, scale=obs_scale, in_keys=["pixels_transformed"], out_keys=["pixels_transformed"], standard_normal=True))
    
    replay_buffer_transform = Compose(*rb_transforms)
    
    return TensorDictReplayBuffer(storage=LazyMemmapStorage(max_size=cfg['replay_buffer']['max_size']), transform=replay_buffer_transform)


def train(experiment: Experiment, train_collector: DataCollectorBase, rb: ReplayBuffer, max_planning_horizon_scheduler: MaxPlanningHorizonScheduler, cfg: DictConfig, agent: TdmAgent, exploration_policy, logger):
    train_batch_size = cfg['train']['train_batch_size']
    train_phase_prefix = "train_"
    train_logger = CometMlLogger(experiment=experiment,
                                 base_logger=SimpleLogger(train_phase_prefix=train_phase_prefix))
    
    video_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / Path(cfg['logging']['video_dir'])
    video_logger = CSVLogger(exp_name=cfg['experiment']['name'], log_dir=video_dir, video_format="mp4", video_fps=cfg['logging']['video_fps'])
    recorder = VideoRecorder(logger=video_logger, tag="step")
    
    step = 0
    
    with set_exploration_type(type=ExplorationType.RANDOM):
        for _ in range(cfg['train']['num_episodes']):
            trained = False
            train_collector.reset()
            for t, data in enumerate(train_collector):
                if t >= max_planning_horizon_scheduler.get_max_planning_horizon():
                    break
                
                exploration_policy.step(frames=data.batch_size[0])
                
                step += 1
                
                if step % cfg['logging']['video_log_step_freq'] == 0:
                    logger.info("Logging video...")
                    log_video(experiment, recorder, data, video_dir, train_phase_prefix, step, logger, cfg)
                
                with set_exploration_type(type=ExplorationType.MEAN):
                    step_data_to_save = TensorDict(
                            source={
                                "pixels": data['pixels'],
                                "action": data['action'],
                                "goal_latent": data['goal_latent'],
                                "planning_horizon": data['planning_horizon'],
                                "next": TensorDict(
                                    source={
                                        "pixels": data['next']['pixels'],
                                        "reward": data['next']['reward']
                                    },
                                    batch_size=[cfg['train']['frames_per_batch']]
                                )
                            },
                            batch_size=[cfg['train']['frames_per_batch']]
                        )
                    
                    train_reward_mean = data['next']['reward'].sum(dim=1).mean().item()
                    train_logger.accumulate_step_metric(key='reward', value=train_reward_mean)
                    train_logger.accumulate_episode_metric(key='reward', value=train_reward_mean)
                    
                    rb.extend(step_data_to_save)
                    
                    if len(rb) >= train_batch_size:
                        train_updates_logger = SimpleLogger(train_phase_prefix=train_phase_prefix)
                        for _ in range(cfg['train']['updates_per_step']):
                            train_data = rb.sample(train_batch_size)
                            train_data = relabel_train_data(train_data, max_planning_horizon_scheduler, rb)
                            
                            logger.info("Training...")
                            train_update_metrics = agent.train(train_data)
                            train_updates_logger.accumulate_step_metrics(train_update_metrics)
                            trained = True
                        
                        train_updates_step_metrics = train_updates_logger.compute_step_metrics()
                        train_logger.accumulate_step_metrics(train_updates_step_metrics)
                
                train_logger.compute_step_metrics()
            
            train_logger.compute_episode_metrics()
            
            if trained:
                max_planning_horizon_scheduler.step()


def log_video(experiment, recorder, data, video_dir, train_phase_prefix, step, logger, cfg):
    for obs in data['pixels']:
        recorder._apply_transform(obs)
    recorder.dump()
    video_files = list((video_dir / Path(cfg['experiment']['name']) / Path('videos')).glob("*.mp4"))
    video_files.sort(reverse=True)
    video_file = video_files[0]
    logger.info(f"Logging {video_file.name} to CometML...")
    experiment.log_video(file=video_file, name=f"{train_phase_prefix}{step}", step=step)


def relabel_train_data(train_data: TensorDict, max_planning_horizon_scheduler: MaxPlanningHorizonScheduler, rb: ReplayBuffer) -> TensorDict:
    planning_horizon_shape = train_data['planning_horizon'].shape
    new_planning_horizon = torch.randint(low=0, high=max_planning_horizon_scheduler.get_max_planning_horizon() + 1, size=planning_horizon_shape)
    train_data['planning_horizon'] = new_planning_horizon
    
    new_goal_latent = rb.sample(batch_size=train_data.batch_size[0])['goal_latent']
    train_data['goal_latent'] = new_goal_latent
    
    return train_data


if __name__ == "__main__":
    main()
