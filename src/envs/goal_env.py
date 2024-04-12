import torch
from torchrl.envs import EnvBase
from typing import Optional
from torchrl.data import BoundedTensorSpec
from tensordict.nn import TensorDictModule
from tensordict import TensorDict


class GoalEnv(EnvBase):
    def __init__(self, 
                 env,
                 raw_obs_height: int, 
                 raw_obs_width: int,
                 env_goal_strategy,
                 encoder_decoder_model: TensorDictModule = None,
                 goal_norm_type: str = 'l1'):
        super().__init__(device=env.device, batch_size=env.batch_size)
        self.env = env
        self.observation_spec = env.observation_spec.clone()
        self.action_spec = env.action_spec.clone()
        self.done_spec = env.done_spec.clone()
        self.state_spec = env.state_spec.clone()
        self.reward_spec = env.reward_spec.clone()
        self.observation_spec['goal_pixels'] = BoundedTensorSpec(
            low=0,
            high=255,
            shape=(raw_obs_width, raw_obs_height, 3),
            device=env.device,
            dtype=torch.uint8
        )
        self.full_state_spec['goal_pixels'] = self.observation_spec['goal_pixels'].clone()
        self.env_goal_strategy = env_goal_strategy
        self.encoder_decoder_model = encoder_decoder_model
        self.goal_norm_type = goal_norm_type
    
    def _step(self, tensordict):
        goal_pixels = tensordict.get("goal_pixels")
        goal_latent = tensordict.get("goal_latent")
        tensordict = self.env._step(tensordict).set("goal_pixels", goal_pixels)
        if self.encoder_decoder_model is not None:
            tensordict.set("goal_latent", goal_latent)
            reward = self.compute_reward(tensordict)
            tensordict.set("reward", reward)
        return tensordict

    def compute_reward(self, tensordict) -> torch.Tensor:
        obs_latent = self.encoder_to_latent_representation(image=tensordict['pixels_transformed'], batch_size=tensordict.batch_size)
        goal_latent = tensordict.get("goal_latent")
        
        if self.goal_norm_type == 'l1':
            reward = torch.abs(obs_latent - goal_latent)
        elif self.goal_norm_type == 'l2':
            reward = torch.sqrt(torch.pow(obs_latent - goal_latent, exponent=2))
        else:
            raise ValueError(f"Unknown goal norm type '{self.goal_norm_type}'")
        
        return reward
        
    def _reset(self, tensordict):
        tensordict = self.env._reset(tensordict)
        tensordict, goal_pixels = self.env_goal_strategy.get_goal_pixels(self.env, tensordict)
        self.goal_pixels = goal_pixels
        tensordict["goal_pixels"] = goal_pixels
        tensordict['next']['goal_pixels'] = goal_pixels
        
        if self.encoder_decoder_model is not None:
            goal_latent = self.encoder_to_latent_representation(image=self.goal_pixels, batch_size=tensordict.batch_size)
            tensordict["goal_latent"] = goal_latent
            tensordict['next']['goal_latent'] = goal_latent
        
        return tensordict
    
    def encoder_to_latent_representation(self, image: torch.Tensor, batch_size: int) -> torch.Tensor:
        input = TensorDict(
            source={
                "image": image.to(self.encoder_decoder_model.device)
            },
            batch_size=batch_size
        )
        return self.encoder_decoder_model(input)['q_z'].loc
        
    
    def _set_seed(self, seed: Optional[int]):
        self.env._set_seed(seed)
