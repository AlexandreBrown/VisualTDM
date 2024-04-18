import torch
from tensordict import TensorDict
from actors.tdm_actor import TdmTd3Actor
from critics.tdm_critic import TdmTd3Critic

class TdmTd3Agent():
    def __init__(self,
                 actor_model_type: str,
                 actor_hidden_layers_out_features: list,
                 actor_hidden_activation_function_name: str,
                 actor_output_activation_function_name: str,
                 actor_learning_rate: float,
                 critic_model_type: str,
                 critic_hidden_layers_out_features: list,
                 critic_use_batch_norm: bool,
                 critic_hidden_activation_function_name: str,
                 critic_output_activation_function_name: str,
                 critic_learning_rate: float,
                 critic_relative: bool,
                 obs_dim: int,
                 actions_dim: int,
                 action_scale: float,
                 action_bias: float,
                 goal_dim: int,
                 goal_latent_dim: int,
                 device: torch.device,
                 polyak_avg: float,
                 norm_type: str,
                 target_update_freq: int,
                 target_policy_action_noise_clip: float,
                 target_policy_action_noise_std: float,
                 state_dim: int,
                 actor_in_keys: int,
                 critic_in_keys: int,
                 action_space_low: torch.Tensor,
                 action_space_high: torch.Tensor):
        self.actor = TdmTd3Actor(model_type=actor_model_type,
                              obs_dim=obs_dim,
                              actions_dim=actions_dim,
                              goal_dim=goal_dim,
                              goal_latent_dim=goal_latent_dim,
                              hidden_layers_out_features=actor_hidden_layers_out_features,
                              hidden_activation_function_name=actor_hidden_activation_function_name,
                              output_activation_function_name=actor_output_activation_function_name,
                              device=device,
                              learning_rate=actor_learning_rate,
                              polyak_avg=polyak_avg,
                              action_scale=action_scale,
                              action_bias=action_bias,
                              state_dim=state_dim,
                              actor_in_keys=actor_in_keys,
                              critic_in_keys=critic_in_keys)
        self.critic = TdmTd3Critic(model_type=critic_model_type,
                                obs_dim=obs_dim,
                                actions_dim=actions_dim,
                                goal_dim=goal_dim,
                                goal_latent_dim=goal_latent_dim,
                                hidden_layers_out_features=critic_hidden_layers_out_features,
                                use_batch_norm=critic_use_batch_norm,
                                hidden_activation_function_name=critic_hidden_activation_function_name,
                                output_activation_function_name=critic_output_activation_function_name,
                                device=device,
                                norm_type=norm_type,
                                actor=self.actor,
                                learning_rate=critic_learning_rate,
                                polyak_avg=polyak_avg,
                                target_policy_action_noise_clip=target_policy_action_noise_clip,
                                target_policy_action_noise_std=target_policy_action_noise_std,
                                state_dim=state_dim,
                                actor_in_keys=actor_in_keys,
                                critic_in_keys=critic_in_keys,
                                action_space_low=action_space_low,
                                action_space_high=action_space_high,
                                relative=critic_relative)
        self.target_update_freq = target_update_freq
        self.num_param_updates = 0
        self.device = device
    
    def train(self, train_data: TensorDict) -> dict:
        train_data = train_data.to(self.device)
        
        critic_logs = self.critic.update(train_data)
        actor_logs = {}
        
        if self.num_param_updates % self.target_update_freq == 0:
            actor_logs = self.actor.update(train_data, self.critic)
            self.critic.update_target_network()
        
        self.num_param_updates += 1
        
        return {**critic_logs, **actor_logs}
