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
from critics.tdm_critic import TdmTd3Critic
from models.vae.model import VAEDecoder, VAEModel
from envs.max_planning_horizon_scheduler import TdmMaxPlanningHorizonScheduler
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
from torchrl.modules import AdditiveGaussianWrapper
from torchrl.envs import EnvBase
from rewards.distance import compute_distance
from tensor_utils import get_tensor
from torchrl.data import SliceSamplerWithoutReplacement

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
 
    eval_policy = TensorDictModule(tdm_agent.actor, in_keys="actor_inputs", out_keys=["action"])
    
    exploration_policy = AdditiveGaussianWrapper(policy=eval_policy, 
                                                 sigma_init=cfg['train']['noise_sigma_init'],
                                                 sigma_end=cfg['train']['noise_sigma_end'],
                                                 annealing_num_steps=cfg['train']['noise_annealing_steps'],
                                                 mean=cfg['train']['noise_mean'],
                                                 std=cfg['train']['noise_std'],
                                                 action_key=eval_policy.out_keys[0],
                                                 spec=train_env.action_spec,
                                                 safe=True)
    
    train_collector = SyncDataCollector(
        create_env_fn=train_env,
        policy=exploration_policy,
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
  
    try:
        train(experiment, train_collector, replay_buffer, tdm_max_planning_horizon_scheduler, cfg, tdm_agent, exploration_policy, logger, eval_env, encoder_decoder_model)
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
    return TensorDictReplayBuffer(storage=LazyMemmapStorage(max_size=cfg['replay_buffer']['max_size']),
                                    sampler=SliceSamplerWithoutReplacement(traj_key="traj", num_slices=cfg['train']['num_trajs'], strict_length=False))


def train(experiment: Experiment, train_collector: DataCollectorBase, replay_buffer: ReplayBuffer, tdm_max_planning_horizon_scheduler: TdmMaxPlanningHorizonScheduler, cfg: DictConfig, agent: TdmTd3Agent, policy, logger, eval_env: TransformedEnv, encoder_decoder_model):
    logger.info("Starting training...")
    train_stage_prefix = "train_"
    eval_stage_prefix = "eval_"
    trained = False
    train_logger = CometMlLogger(experiment=experiment,
                                 base_logger=SimpleLogger(stage_prefix=train_stage_prefix))
    
    video_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / Path(cfg['logging']['video_dir'])
    video_logger = CSVLogger(exp_name=cfg['experiment']['name'], log_dir=video_dir, video_format="mp4", video_fps=cfg['logging']['video_fps'])
    recorder = VideoRecorder(logger=video_logger, tag="step", skip=1)
    
    global_traj_id = 0
    global_step = 0
    last_traj_id = 0
    for data in train_collector:
        traj_ids = data['collector']['traj_ids'].unique()
        for batch_traj_id in traj_ids:
            if last_traj_id != batch_traj_id.item():
                global_traj_id += 1
            
            traj_data = data[data['collector']['traj_ids'] == batch_traj_id]
        
            step_data = get_step_data_of_interest(data=traj_data, cfg=cfg)
            step_data['traj'] = torch.full(size=(step_data.shape[0],), fill_value=global_traj_id, dtype=torch.int32)
            
            replay_buffer.extend(step_data)
            
            if last_traj_id == batch_traj_id.item() and torch.nonzero(traj_data['done']).numel() > 0:
                global_traj_id += 1
            
            last_traj_id = batch_traj_id.item()
        
        train_logger.should_log = can_train(replay_buffer, cfg, cfg['train']['batch_size'], step=global_step)
            
        accumulate_train_metrics(data, train_logger, tdm_max_planning_horizon_scheduler)
        
        trained = do_train_updates(replay_buffer, cfg, global_step, train_stage_prefix, agent, train_logger, tdm_max_planning_horizon_scheduler)
        
        log_video(experiment, recorder, video_dir, eval_stage_prefix, global_step, logger, cfg, eval_env, policy, encoder_decoder_model, trained)
        log_q_values(experiment, eval_stage_prefix, eval_env, policy, encoder_decoder_model, trained, cfg, global_step, agent.critic, tdm_max_planning_horizon_scheduler)
    
        train_logger.compute_step_metrics()
        policy.step(cfg['env']['frames_per_batch'])
        global_step += 1

        tdm_max_planning_horizon_scheduler.step(trained)


def accumulate_train_metrics(data: TensorDict, train_logger: CometMlLogger, tdm_max_planning_horizon_scheduler: TdmMaxPlanningHorizonScheduler):
    train_reward_mean = data['next']['reward'].sum(dim=1).mean().item()
    train_logger.accumulate_step_metric(key='reward', value=train_reward_mean)
    train_logger.accumulate_episode_metric(key='reward', value=train_reward_mean)
    
    if 'goal_latent_reached' in data.keys():
        goal_latent_reached_rate = data['goal_latent_reached'].type(torch.float32).mean().item()
        train_logger.accumulate_step_metric(key='goal_latent_reached', value=goal_latent_reached_rate)
        train_logger.accumulate_episode_metric(key='goal_latent_reached', value=goal_latent_reached_rate)
    
    if 'goal_latent_l2_distance' in data.keys():
        goal_latent_l2_distance_mean = data['goal_latent_l2_distance'].mean().item()
        train_logger.accumulate_step_metric(key='goal_latent_l2_distance', value=goal_latent_l2_distance_mean)
        train_logger.accumulate_episode_metric(key='goal_latent_l2_distance', value=goal_latent_l2_distance_mean)
    
    if 'goal_latent_l1_distance' in data.keys():
        goal_latent_l1_distance_mean = data['goal_latent_l1_distance'].mean().item()
        train_logger.accumulate_step_metric(key='goal_latent_l1_distance', value=goal_latent_l1_distance_mean)
        train_logger.accumulate_episode_metric(key='goal_latent_l1_distance', value=goal_latent_l1_distance_mean)
    
    if 'goal_latent_cosine_distance' in data.keys():
        goal_latent_cosine_distance_mean = data['goal_latent_cosine_distance'].mean().item()
        train_logger.accumulate_step_metric(key='goal_latent_cosine_distance', value=goal_latent_cosine_distance_mean)
        train_logger.accumulate_episode_metric(key='goal_latent_cosine_distance', value=goal_latent_cosine_distance_mean)
    
    if 'original_reward' in data['next'].keys():
        original_reward_mean = -data['next']['original_reward'].mean().item()
        train_logger.accumulate_step_metric(key='actual_distance_to_goal', value=original_reward_mean)
        train_logger.accumulate_episode_metric(key='actual_distance_to_goal', value=original_reward_mean)
    
    train_logger.log_step_metric(key='max_planning_horizon', value=tdm_max_planning_horizon_scheduler.get_max_planning_horizon())


def get_step_data_of_interest(data: TensorDict, cfg: DictConfig) -> TensorDict:
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
        batch_size=[data.shape[0]]
    )
    
    return TensorDict(
        source=data_time_t,
        batch_size=[data.shape[0]]
    )


def do_train_updates(replay_buffer: ReplayBuffer, cfg: DictConfig, step: int, stage_prefix: str, agent, train_logger, tdm_max_planning_horizon_scheduler: TdmMaxPlanningHorizonScheduler) -> bool:
    train_batch_size = cfg['train']['batch_size']
    
    if can_train(replay_buffer, cfg, train_batch_size, step):
        train_updates_logger = SimpleLogger(stage_prefix=stage_prefix)
        
        for _ in range(cfg['train']['updates_per_step']):
            train_update_metrics = do_train_update(agent, replay_buffer, train_batch_size, tdm_max_planning_horizon_scheduler, cfg)
            train_updates_logger.accumulate_step_metrics(train_update_metrics)
        
        trained = True
        train_updates_step_metrics = train_updates_logger.compute_step_metrics()
        train_logger.accumulate_step_metrics(train_updates_step_metrics)
    else:
        trained = False
        
    return trained


def can_train(replay_buffer: ReplayBuffer, cfg: DictConfig, train_batch_size: int, step: int) -> bool:
    is_random_exploration_over = get_is_random_exploration_over(replay_buffer, cfg)
    is_learning_step = step % cfg['train']['learning_frequency'] == 0
    can_sample_train_batch = len(replay_buffer) >= train_batch_size
    
    return is_random_exploration_over and is_learning_step and can_sample_train_batch


def get_is_random_exploration_over(replay_buffer: ReplayBuffer, cfg: DictConfig) -> bool:
    return len(replay_buffer) >= cfg['env']['init_random_frames']


def do_train_update(agent: TdmTd3Agent, replay_buffer: ReplayBuffer, train_batch_size: int, tdm_max_planning_horizon_scheduler: TdmMaxPlanningHorizonScheduler, cfg: DictConfig):
    train_data_sample = replay_buffer.sample(train_batch_size)
    
    train_data_sample = relabel_train_data(train_data_sample, tdm_max_planning_horizon_scheduler, replay_buffer)
    
    train_update_metrics = agent.train(train_data_sample)
    
    return train_update_metrics


def relabel_train_data(train_data_sample: TensorDict, tdm_max_planning_horizon_scheduler: TdmMaxPlanningHorizonScheduler, replay_buffer: ReplayBuffer) -> TensorDict:
    train_data_sample_relabeled = train_data_sample.clone(recurse=True)

    train_data_sample_relabeled = relabel_planning_horizon(train_data_sample_relabeled, tdm_max_planning_horizon_scheduler)
    train_data_sample_relabeled = relabel_goal(train_data_sample_relabeled)
    
    return train_data_sample_relabeled


def relabel_planning_horizon(train_data_sample_relabeled: TensorDict, tdm_max_planning_horizon_scheduler: TdmMaxPlanningHorizonScheduler) -> TensorDict:
    batch_size = train_data_sample_relabeled.batch_size[0]
    
    new_planning_horizon = torch.randint(low=0, high=tdm_max_planning_horizon_scheduler.get_max_planning_horizon() + 1, size=(batch_size, 1))
    
    train_data_sample_relabeled['planning_horizon'] = new_planning_horizon
    
    return train_data_sample_relabeled


def relabel_goal(train_data_sample_relabeled: TensorDict) -> TensorDict:
    relabel_index = -1
    train_data_sample_goal_relabeled = train_data_sample_relabeled.clone(recurse=True)
    for traj_id in torch.unique_consecutive(train_data_sample_relabeled['traj']):
        traj_data = train_data_sample_relabeled[train_data_sample_relabeled['traj'] == traj_id]
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

        relabel_index += 1
    
    return train_data_sample_goal_relabeled


def log_video(experiment: Experiment, recorder: VideoRecorder, video_dir: Path, stage_prefix: str, step: int, logger, cfg: DictConfig, eval_env: EnvBase, eval_policy: TensorDictModule, encoder_decoder_model: VAEModel, trained: bool):
    if not can_log_video(step, cfg, trained):
        return
    
    with set_exploration_type(type=ExplorationType.MEAN):
        rollout_data = eval_env.rollout(max_steps=cfg['env']['max_frames_per_traj'], policy=eval_policy, break_when_any_done=False)

        decoder = encoder_decoder_model.decoder
        
        log_step = step+1
        
        log_goal_image(experiment, rollout_data, decoder, log_step, stage_prefix)
        log_obs_images(experiment, rollout_data, decoder, log_step, stage_prefix, cfg)
        
        for obs in rollout_data['pixels']:
            recorder._apply_transform(obs)
        
        if len(recorder.obs) < cfg['logging']['video_frames']:
            return
        
        recorder.dump()
        video_files = list((video_dir / Path(cfg['experiment']['name']) / Path('videos')).glob("*.mp4"))
        video_files.sort(reverse=True)
        video_file = video_files[0]
        logger.info(f"Logging video {video_file.name} to CometML...")
        experiment.log_video(file=video_file, name=f"{stage_prefix}rollout_{log_step}", step=log_step)


def can_log_video(step: int, cfg: DictConfig, trained: bool) -> bool:
    return trained and step % cfg['logging']['video_log_step_freq'] == 0


def log_goal_image(experiment: Experiment, rollout_data: TensorDict, decoder: VAEDecoder, step: int, stage_prefix: str):
    goal_latent = rollout_data['goal_latent'][0].unsqueeze(0)
    goal_rgb_image = decode_to_rgb(decoder, goal_latent)
    experiment.log_image(image_data=goal_rgb_image, name=f"{stage_prefix}decoded_goal_{step}", image_channels='first', step=step)
    actual_goal_rgb_image = rollout_data['goal_pixels'][0]
    experiment.log_image(image_data=actual_goal_rgb_image, name=f"{stage_prefix}actual_goal_{step}", image_channels='first', step=step)


def log_obs_images(experiment: Experiment, rollout_data: TensorDict, decoder: VAEDecoder, step: int, stage_prefix: str, cfg: DictConfig):
    goal_latent = rollout_data['goal_latent'][0].unsqueeze(0)
    
    traj_step_index_before_done = get_traj_step_index_before_done(rollout_data)
    
    distance_type = cfg['train']['reward_distance_type']
    
    pixels_latent = rollout_data['pixels_latent'][0].unsqueeze(0)
    pixels_rgb_image = decode_to_rgb(decoder, pixels_latent)
    obs_distance_to_goal_latent = compute_distance(distance_type=distance_type,
                                               obs_latent=pixels_latent,
                                               goal_latent=goal_latent).mean()
    experiment.log_image(image_data=pixels_rgb_image, name=f"{stage_prefix}decoded_obs_0_{step}", image_channels='first', step=step, metadata={
        'traj_step': 0,
        f"{distance_type}_distance_to_goal_latent": obs_distance_to_goal_latent
    })
    
    if traj_step_index_before_done < 3:
        return
    
    random_index = torch.randint(low=1, high=traj_step_index_before_done - 1, size=(1,)).item()
    random_pixels_latent = rollout_data['pixels_latent'][random_index].unsqueeze(0)
    random_pixels_rgb_image = decode_to_rgb(decoder, random_pixels_latent)
    random_obs_distance_to_goal_latent = compute_distance(distance_type=distance_type,
                                               obs_latent=random_pixels_latent,
                                               goal_latent=goal_latent).mean()
    experiment.log_image(image_data=random_pixels_rgb_image, name=f"{stage_prefix}decoded_obs_{random_index}_{step}", image_channels='first', step=step, metadata={
        'traj_step': random_index,
        f"{distance_type}_distance_to_goal_latent": random_obs_distance_to_goal_latent
    })
    
    last_pixels_latent = rollout_data['pixels_latent'][traj_step_index_before_done].unsqueeze(0)
    last_pixels_rgb_image = decode_to_rgb(decoder, last_pixels_latent)
    last_obs_distance_to_goal_latent = compute_distance(distance_type=distance_type,
                                               obs_latent=last_pixels_latent,
                                               goal_latent=goal_latent).mean()
    experiment.log_image(image_data=last_pixels_rgb_image, name=f"{stage_prefix}decoded_obs_{traj_step_index_before_done}_{step}", image_channels='first', step=step, metadata={
        'traj_step': traj_step_index_before_done,
        f"{distance_type}_distance_to_goal_latent": last_obs_distance_to_goal_latent
    }) 


def get_traj_step_index_before_done(data: TensorDict) -> int:
    first_done = torch.nonzero(data['done'].squeeze(1))
    if first_done.numel() == 0:
        traj_last_frame_index = data.shape[0] - 1
    else:
        traj_last_frame_index = first_done[0].item() - 1
    
    return traj_last_frame_index


def log_q_values(experiment: Experiment, stage_prefix: str, eval_env: EnvBase, eval_policy: TensorDictModule, encoder_decoder_model: VAEModel, trained: bool, cfg: DictConfig, step: int, critic: TdmTd3Critic, tdm_max_planning_horizon_scheduler: TdmMaxPlanningHorizonScheduler):
    if not (trained and step % cfg['logging']['images_log_step_freq'] == 0):
        return
    
    with set_exploration_type(type=ExplorationType.MEAN):
        traj_data = eval_env.rollout(max_steps=cfg['env']['max_frames_per_traj'], policy=eval_policy)
        
        attempts = 0
        while traj_data.batch_size[0] < 16:
            traj_data = eval_env.rollout(max_steps=cfg['env']['max_frames_per_traj'], policy=eval_policy)
            attempts += 1
            if attempts == 10:
                return
        
        traj_data_0 = traj_data[0].select(*critic.critic_in_keys)
        traj_data_1 = traj_data[1].select(*critic.critic_in_keys)
        traj_data_1['pixels_transformed'] = traj_data['pixels_transformed'][1]
        traj_data_5 = traj_data[5].select(*critic.critic_in_keys)
        traj_data_5['pixels_transformed'] = traj_data['pixels_transformed'][5]
        traj_data_10 = traj_data[10].select(*critic.critic_in_keys)
        traj_data_10['pixels_transformed'] = traj_data['pixels_transformed'][10]
        traj_data_15 = traj_data[15].select(*critic.critic_in_keys)
        traj_data_15['pixels_transformed'] = traj_data['pixels_transformed'][15]
        
        goal_rgb_image = traj_data['goal_pixels'][0]
        experiment.log_image(image_data=goal_rgb_image, name=f"{stage_prefix}critic_actual_goal_{step}", image_channels='first', step=step)
        log_q_functions_preds(experiment, stage_prefix, traj_data_0, traj_data_1, critic, encoder_decoder_model, step, tdm_max_planning_horizon_scheduler, pred_tdm_planning_horizon=1)
        log_q_functions_preds(experiment, stage_prefix, traj_data_0, traj_data_5, critic, encoder_decoder_model, step, tdm_max_planning_horizon_scheduler, pred_tdm_planning_horizon=5)
        log_q_functions_preds(experiment, stage_prefix, traj_data_0, traj_data_10, critic, encoder_decoder_model, step, tdm_max_planning_horizon_scheduler, pred_tdm_planning_horizon=10)
        log_q_functions_preds(experiment, stage_prefix, traj_data_0, traj_data_15, critic, encoder_decoder_model, step, tdm_max_planning_horizon_scheduler, pred_tdm_planning_horizon=15)


def log_q_functions_preds(experiment: Experiment, stage_prefix: str, step_data: TensorDict, expected_step_data: TensorDict, critic: TdmTd3Critic, encoder_decoder_model: VAEModel, step: int, tdm_max_planning_horizon_scheduler: TdmMaxPlanningHorizonScheduler, pred_tdm_planning_horizon: int):
    step_data['planning_horizon'] = torch.ones(size=(1,)) * pred_tdm_planning_horizon
    qf_input = get_tensor(step_data, keys=critic.critic_in_keys).unsqueeze(0)
    critic.qf1.eval()
    q_value, predicted_latent_state = critic.qf1.forward(qf_input, output_predicted_latent_state=True)
    critic.qf1.train()
    
    logging_step = step + 1
    tdm_max_planning_horizon = tdm_max_planning_horizon_scheduler.get_max_planning_horizon()
    
    if critic.qf1.reward_dim != 1:
        predicted_pixels_latent = predicted_latent_state.unsqueeze(0)
        predicted_pixels_rgb_image = decode_to_rgb(encoder_decoder_model.decoder, predicted_pixels_latent)
    
        experiment.log_image(image_data=predicted_pixels_rgb_image, name=f"{stage_prefix}critic_predicted_obs_tau_{pred_tdm_planning_horizon}", image_channels='first', step=logging_step, metadata={
            'q_value_mean': q_value.mean().item(),
            'tdm_max_planning_horizon': tdm_max_planning_horizon,
            'tdm_prediction_planning_horizon': pred_tdm_planning_horizon
        })
    
    decoded_next_pixels_latent = expected_step_data['pixels_latent'].unsqueeze(0)
    decoded_next_pixels_rgb_image = decode_to_rgb(encoder_decoder_model.decoder, decoded_next_pixels_latent)
    
    actual_next_pixels_rgb_image = expected_step_data['pixels_transformed']
    
    experiment.log_image(image_data=decoded_next_pixels_rgb_image, name=f"{stage_prefix}critic_decoded_next_obs_tau_{pred_tdm_planning_horizon}", image_channels='first', step=logging_step, metadata={
        'q_value_mean': q_value.mean().item(),
        'tdm_max_planning_horizon': tdm_max_planning_horizon,
        'tdm_prediction_planning_horizon': pred_tdm_planning_horizon
    })
    experiment.log_image(image_data=actual_next_pixels_rgb_image, name=f"{stage_prefix}critic_actual_next_obs_tau_{pred_tdm_planning_horizon}", image_channels='first', step=logging_step, metadata={
        'q_value_mean': q_value.mean().item(),
        'tdm_max_planning_horizon': tdm_max_planning_horizon,
        'tdm_prediction_planning_horizon': pred_tdm_planning_horizon
    })

if __name__ == "__main__":
    main()
