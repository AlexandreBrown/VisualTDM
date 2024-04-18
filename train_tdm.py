from comet_ml import Experiment
from comet_ml.exceptions import InterruptedExperiment
from comet_ml import API
import hydra
import logging
import torch
import os
from pathlib import Path
from omegaconf import DictConfig
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.collectors.collectors import DataCollectorBase
from torchrl.data.replay_buffers import LazyMemmapStorage
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.transforms import ExcludeTransform
from tensordict.nn import TensorDictModule
from agents.tdm_agent import TdmTd3Agent
from torchrl.modules.tensordict_module import AdditiveGaussianWrapper
from models.vae.model import VAEDecoder, VAEModel
from envs.max_planning_horizon_scheduler import MaxPlanningHorizonScheduler
from torchrl.envs.utils import set_exploration_type
from torchrl.envs.utils import ExplorationType
from tensordict import TensorDict
from loggers.simple_logger import SimpleLogger
from loggers.cometml_logger import CometMlLogger
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record import VideoRecorder
from torchrl.envs import TransformedEnv
from envs.tdm_env_factory import create_tdm_env
from models.vae.utils import decode_to_rgb

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

    max_planning_horizon_scheduler = MaxPlanningHorizonScheduler(initial_max_planning_horizon=cfg['train']['initial_max_planning_horizon'],
                                                          steps_per_traj=cfg['train']['max_frames_per_traj'],
                                                          n_cycle=cfg['train']['planning_horizon_annealing_cycles'],
                                                          ratio=cfg['train']['planning_horizon_annealing_ratio'],
                                                          enable=cfg['train']['planning_horizon_annealing'])

    train_env = create_tdm_env(cfg, encoder_decoder_model, max_planning_horizon_scheduler)
    eval_env = create_tdm_env(cfg, encoder_decoder_model, max_planning_horizon_scheduler)
    
    actions_dim = train_env.action_spec.shape[0]
    action_space_low = train_env.action_spec.space.low
    action_space_high = train_env.action_spec.space.high
    action_scale = (action_space_high - action_space_low) / 2
    action_bias = (action_space_low + action_space_high) / 2
    actor_params = cfg['models']['actor']
    critic_params = cfg['models']['critic']
    state_dim = train_env.observation_spec['observation'].shape[0]
    if "desired_goal" in train_env.observation_spec.keys():
        goal_dim = train_env.observation_spec['desired_goal'].shape[0]
    else:
        goal_dim = None
    tdm_agent = TdmTd3Agent(actor_model_type=actor_params['model_type'],
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
                         obs_dim=cfg['env']['obs']['dim'],
                         actions_dim=actions_dim,
                         action_scale=action_scale,
                         action_bias=action_bias,
                         goal_dim=goal_dim,
                         goal_latent_dim=cfg['env']['goal']['latent_dim'],
                         device=models_device,
                         polyak_avg=cfg['train']['polyak_avg'],
                         norm_type=cfg['train']['reward_norm_type'],
                         target_update_freq=cfg['train']['target_update_freq'],
                         target_policy_action_noise_clip=cfg['train']['target_policy_action_noise_clip'],
                         target_policy_action_noise_std=cfg['train']['target_policy_action_noise_std'],
                         state_dim=state_dim,
                         actor_in_keys=list(cfg['models']['actor']['in_keys']),
                         critic_in_keys=list(cfg['models']['critic']['in_keys']),
                         action_space_low=action_space_low,
                         action_space_high=action_space_high)
 
    policy = TensorDictModule(tdm_agent.actor, in_keys="actor_inputs", out_keys=["action"])
    
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
    
    rb = create_replay_buffer(cfg)
  
    try:
        train(experiment, train_collector, rb, max_planning_horizon_scheduler, cfg, tdm_agent, exploration_policy, logger, eval_env, encoder_decoder_model)
    except InterruptedExperiment as exc:
        experiment.log_other("status", str(exc))
        logger.info("Experiment interrupted!")
    
    logger.info("Training done!")
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
    
    for param in encoder_decoder_model.parameters():
        param.requires_grad = False
    
    return encoder_decoder_model


def create_experiment(api_key: str, project_name: str, workspasce: str) -> Experiment:
    return Experiment(
        api_key=api_key,
        project_name=project_name,
        workspace=workspasce
    )


def create_replay_buffer(cfg: DictConfig) -> ReplayBuffer:
    return TensorDictReplayBuffer(storage=LazyMemmapStorage(max_size=cfg['replay_buffer']['max_size']))


def train(experiment: Experiment, train_collector: DataCollectorBase, rb: ReplayBuffer, max_planning_horizon_scheduler: MaxPlanningHorizonScheduler, cfg: DictConfig, agent: TdmTd3Agent, exploration_policy, logger, eval_env: TransformedEnv, encoder_decoder_model):
    logger.info("Starting training...")
    train_batch_size = cfg['train']['train_batch_size']
    train_phase_prefix = "train_"
    eval_phase_prefix = "eval_"
    train_logger = CometMlLogger(experiment=experiment,
                                 base_logger=SimpleLogger(stage_prefix=train_phase_prefix))
    
    video_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / Path(cfg['logging']['video_dir'])
    video_logger = CSVLogger(exp_name=cfg['experiment']['name'], log_dir=video_dir, video_format="mp4", video_fps=cfg['logging']['video_fps'])
    recorder = VideoRecorder(logger=video_logger, tag="step", skip=1)
    
    step = 0
    
    with set_exploration_type(type=ExplorationType.RANDOM):
        for _ in range(cfg['train']['num_episodes']):
            trained = False
            train_collector.reset()
            for t, data in enumerate(train_collector):
                if t >= max_planning_horizon_scheduler.get_max_planning_horizon():
                    break
                    
                if step % cfg['logging']['video_log_step_freq'] == 0:
                    with set_exploration_type(type=ExplorationType.MEAN):
                        log_video(experiment, recorder, video_dir, eval_phase_prefix, step+1, logger, cfg, eval_env, exploration_policy, encoder_decoder_model)
                
                exploration_policy.step(frames=data.batch_size[0])
                step += 1
                
                train_reward_mean = data['next']['reward'].sum(dim=1).mean().item()
                train_logger.accumulate_step_metric(key='reward', value=train_reward_mean)
                train_logger.accumulate_episode_metric(key='reward', value=train_reward_mean)
                
                goal_reached_rate = data['goal_reached'].type(torch.float32).mean().item()
                train_logger.accumulate_step_metric(key='goal_reached', value=goal_reached_rate)
                train_logger.accumulate_episode_metric(key='goal_reached', value=goal_reached_rate)
                
                goal_l2_distance_mean = data['goal_l2_distance'].mean().item()
                train_logger.accumulate_step_metric(key='goal_l2_distance', value=goal_l2_distance_mean)
                train_logger.accumulate_episode_metric(key='goal_l2_distance', value=goal_l2_distance_mean)
                
                for planning_horizon in data['planning_horizon']:
                    train_logger.log_step_metric(key='planning_horizon', value=planning_horizon.item())
                
                step_data_to_save = get_step_data_to_save(data=data, cfg=cfg)
                
                rb.extend(step_data_to_save)
                
                if len(rb) >= train_batch_size:
                    train_updates_logger = SimpleLogger(stage_prefix=train_phase_prefix)
                    for _ in range(cfg['train']['updates_per_step']):
                        train_data_sample = rb.sample(train_batch_size)
                        train_data_sample = relabel_train_data(train_data_sample, max_planning_horizon_scheduler, rb)
                        
                        train_update_metrics = agent.train(train_data_sample)
                        train_updates_logger.accumulate_step_metrics(train_update_metrics)
                        trained = True
                    
                    train_updates_step_metrics = train_updates_logger.compute_step_metrics()
                    train_logger.accumulate_step_metrics(train_updates_step_metrics)
            
                train_logger.compute_step_metrics()
            
            train_logger.compute_episode_metrics()
            
            if trained:
                max_planning_horizon_scheduler.step()


def get_step_data_to_save(data: TensorDict, cfg: DictConfig):
    data_time_t = {}
    keys_of_interest = set(cfg['env']['keys_of_interest'])
    
    for key in keys_of_interest:
        if key in data.keys():
            data_time_t[key] = data[key]
        
    data_time_t_plus_1 = {}
    for key in keys_of_interest:
        if key in data['next'].keys():
            data_time_t_plus_1[key] = data['next'][key]
    
    data_time_t['next'] = TensorDict(
        source=data_time_t_plus_1,
        batch_size=[cfg['train']['frames_per_batch']]
    )
    
    return TensorDict(
        source=data_time_t,
        batch_size=[cfg['train']['frames_per_batch']]
    )


def log_video(experiment: Experiment, recorder: VideoRecorder, video_dir, stage_prefix, step, logger, cfg, eval_env: TransformedEnv, policy, encoder_decoder_model):
    rollout_data = eval_env.rollout(max_steps=cfg['logging']['video_frames'], policy=policy, break_when_any_done=False)
    
    decoder = encoder_decoder_model.decoder
    
    log_goal_image(experiment, rollout_data, decoder, step, stage_prefix)
    log_obs_image(experiment, rollout_data, decoder, step, stage_prefix)
    
    for obs in rollout_data['pixels']:
        recorder._apply_transform(obs)
    
    if len(recorder.obs) < cfg['logging']['video_frames']:
        return
    
    recorder.dump()
    video_files = list((video_dir / Path(cfg['experiment']['name']) / Path('videos')).glob("*.mp4"))
    video_files.sort(reverse=True)
    video_file = video_files[0]
    logger.info(f"Logging {video_file.name} to CometML...")
    experiment.log_video(file=video_file, name=f"{stage_prefix}rollout_{step}", step=step)


def log_goal_image(experiment: Experiment, rollout_data: TensorDict, decoder: VAEDecoder, step: int, stage_prefix: str):
    goal_latent = rollout_data['goal_latent'][0].unsqueeze(0)
    goal_rgb_image = decode_to_rgb(decoder, goal_latent)
    experiment.log_image(image_data=goal_rgb_image, name=f"{stage_prefix}goal_{step}", image_channels='first', step=step)


def log_obs_image(experiment: Experiment, rollout_data: TensorDict, decoder: VAEDecoder, step: int, stage_prefix: str):
    pixels_latent = rollout_data['pixels_latent'][0].unsqueeze(0)
    pixels_rgb_image = decode_to_rgb(decoder, pixels_latent)
    experiment.log_image(image_data=pixels_rgb_image, name=f"{stage_prefix}obs_{step}", image_channels='first', step=step)


def relabel_train_data(train_data_sample: TensorDict, max_planning_horizon_scheduler: MaxPlanningHorizonScheduler, rb: ReplayBuffer) -> TensorDict:
    
    train_data_sample_relabeled = train_data_sample.clone(recurse=True)
    
    planning_horizon_shape = train_data_sample_relabeled['planning_horizon'].shape
    new_planning_horizon = torch.randint(low=0, high=max_planning_horizon_scheduler.get_max_planning_horizon() + 1, size=planning_horizon_shape)
    train_data_sample_relabeled['planning_horizon'] = new_planning_horizon
    
    new_goal_latent = rb.sample(batch_size=train_data_sample_relabeled.batch_size[0])['goal_latent']
    train_data_sample_relabeled['goal_latent'] = new_goal_latent
    
    return train_data_sample_relabeled


if __name__ == "__main__":
    main()
