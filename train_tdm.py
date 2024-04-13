from comet_ml import Experiment, OfflineExperiment
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
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.transforms import Compose
from torchrl.envs.transforms import ToTensorImage
from torchrl.envs.transforms import Resize
from torchrl.envs.transforms import ExcludeTransform
from torchrl.envs.transforms import ObservationNorm
from tensordict.nn import TensorDictModule
from envs.transforms.add_planning_horizon import AddPlanningHorizon
from envs.transforms.add_goal_latent_representation import AddGoalLatentRepresentation
from models.tdm.policy import TdmPolicy
from torchrl.modules.tensordict_module import AdditiveGaussianWrapper
from models.vae.model import VAEModel
from envs.transforms.compute_latent_goal_distance_vector_reward import ComputeLatentGoalDistanceVectorReward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@hydra.main(version_base=None, config_path="configs/", config_name="tdm_training")
def main(cfg: DictConfig):
    is_cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA AVAILABLE: {is_cuda_available}")
    
    model_device = torch.device(cfg['models']['device'])
    
    COMET_ML_API_KEY = os.getenv("COMET_ML_API_KEY")
    COMET_ML_PROJECT_NAME = os.getenv("COMET_ML_PROJECT_NAME")
    COMET_ML_WORKSPACE = os.getenv("COMET_ML_WORKSPACE")
    
    encoder_decoder_model = download_encoder_decoder_model(api_key=COMET_ML_API_KEY, workspace=COMET_ML_WORKSPACE, cfg=cfg, device=model_device)
    
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
    
    train_env.append_transform(AddPlanningHorizon(initial_max_planning_horizon=cfg['train']['initial_max_planning_horizon']))
    train_env.append_transform(AddGoalLatentRepresentation(encoder_decoder_model=encoder_decoder_model,
                                                           latent_dim=cfg['env']['goal']['latent_dim']))
    train_env.append_transform(ComputeLatentGoalDistanceVectorReward(norm_type=cfg['train']['reward_norm_type'],
                                                                     encoder=encoder_decoder_model,
                                                                     latent_dim=cfg['env']['goal']['latent_dim']))
    
    tdm_policy = TdmPolicy(obs_dim=cfg['env']['obs']['dim'],
                           goal_latent_dim=cfg['env']['goal']['latent_dim'],
                           fc1_out_features=cfg['models']['policy']['fc1_out_features'],
                           actions_dim=train_env.action_spec.shape[0],
                           device=model_device)
    policy = TensorDictModule(tdm_policy, in_keys=["pixels_transformed", "goal_latent", "planning_horizon"], out_keys=["action"])
    
    exploration_policy = AdditiveGaussianWrapper(policy=policy, spec=train_env.action_spec)
    
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
    
    rb = create_replay_buffer(train_env, cfg, encoder_decoder_model)
  
    try:
        train(experiment, train_collector, rb, num_episodes=cfg['train']['num_episodes'])          
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
    return OfflineExperiment(
        api_key=api_key,
        project_name=project_name,
        workspace=workspasce
    )


def create_replay_buffer(train_env, cfg: DictConfig, encoder_decoder_model: TensorDictModule) -> ReplayBuffer:
    rb_transforms = []
    if cfg['env']['obs']['normalize']:
        rb_transforms.append(ToTensorImage(in_keys=["pixels", ("next", "pixels")], out_keys=["pixels_transformed", ("next", "pixels_transformed")]))
        rb_transforms.append(ToTensorImage(in_keys=["goal_pixels", ("next", "goal_pixels")], out_keys=["goal_pixels_transformed", ("next", "goal_pixels_transformed")]))
    rb_transforms.append(Resize(in_keys=["pixels_transformed", ("next", "pixels_transformed")], w=cfg['env']['obs']['width'], h=cfg['env']['obs']['height']))
    rb_transforms.append(Resize(in_keys=["goal_pixels_transformed", ("next", "goal_pixels_transformed")], w=cfg['env']['obs']['width'], h=cfg['env']['obs']['height']))
    
    obs_loc = 0.
    obs_scale = 1.
    
    if cfg['env']['obs']['standardize']:
        obs_loc = train_env.transform[-1].loc
        obs_scale = train_env.transform[-1].scale
        rb_transforms.append(ObservationNorm(loc=obs_loc, scale=obs_scale, in_keys=["pixels_transformed"], out_keys=["pixels_transformed"], standard_normal=True))
        rb_transforms.append(ObservationNorm(loc=obs_loc, scale=obs_scale, in_keys=["goal_pixels_transformed"], out_keys=["goal_pixels_transformed"], standard_normal=True))
    
    rb_transforms.append(AddPlanningHorizon(initial_max_planning_horizon=cfg['train']['initial_max_planning_horizon']))
    rb_transforms.append(AddGoalLatentRepresentation(encoder_decoder_model=encoder_decoder_model,
                                                     latent_dim=cfg['env']['goal']['latent_dim']))
    rb_transforms.append(ComputeLatentGoalDistanceVectorReward(norm_type=cfg['train']['reward_norm_type'],
                                                               encoder=encoder_decoder_model,
                                                               latent_dim=cfg['env']['goal']['latent_dim']))
    
    replay_buffer_transform = Compose(*rb_transforms)
    
    return ReplayBuffer(storage=LazyMemmapStorage(max_size=cfg['replay_buffer']['max_size']), transform=replay_buffer_transform)


def train(experiment: Experiment, train_collector: DataCollectorBase, rb: ReplayBuffer, num_episodes: int):
    with experiment.train():
        for n in range(num_episodes):
            train_collector.reset()
            for t, data in enumerate(train_collector):
                current_goal = data['goal_pixels'].squeeze(0)
                next_goal = data['next']['goal_pixels'].squeeze(0)
                # Store in rb


if __name__ == "__main__":
    main()
