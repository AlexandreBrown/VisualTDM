import torch
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from actors.tdm_actor import TdmActor
from critics.tdm_critic import TdmCritic

class TdmAgent():
    def __init__(self,
                 actor_model_type: str,
                 actor_hidden_layers_out_features: list,
                 actor_hidden_activation_function_name: str,
                 actor_output_activation_function_name: str,
                 actor_learning_rate: float,
                 critic_model_type: str,
                 critic_hidden_layers_out_features: list,
                 critic_hidden_activation_function_name: str,
                 critic_output_activation_function_name: str,
                 critic_learning_rate: float,
                 obs_dim: int,
                 actions_dim: int,
                 action_scale: float,
                 action_bias: float,
                 goal_latent_dim: int,
                 device: torch.device,
                 polyak_avg: float,
                 encoder: TensorDictModule,
                 norm_type: str,
                 target_update_freq: int):
        self.actor = TdmActor(model_type=actor_model_type,
                              obs_dim=obs_dim,
                              actions_dim=actions_dim,
                              goal_latent_dim=goal_latent_dim,
                              hidden_layers_out_features=actor_hidden_layers_out_features,
                              hidden_activation_function_name=actor_hidden_activation_function_name,
                              output_activation_function_name=actor_output_activation_function_name,
                              device=device,
                              learning_rate=actor_learning_rate,
                              polyak_avg=polyak_avg,
                              encoder=encoder,
                              action_scale=action_scale,
                              action_bias=action_bias)
        self.critic = TdmCritic(model_type=critic_model_type,
                                obs_dim=obs_dim,
                                actions_dim=actions_dim,
                                goal_latent_dim=goal_latent_dim,
                                hidden_layers_out_features=critic_hidden_layers_out_features,
                                hidden_activation_function_name=critic_hidden_activation_function_name,
                                output_activation_function_name=critic_output_activation_function_name,
                                device=device,
                                norm_type=norm_type,
                                encoder=encoder,
                                actor=self.actor,
                                learning_rate=critic_learning_rate,
                                polyak_avg=polyak_avg)
        self.target_update_freq = target_update_freq
        self.num_param_updates = 0
        self.device = device
    
    def train(self, train_data: TensorDict) -> dict:
        train_data = train_data.to(self.device)
        
        critic_logs = self.critic.update(train_data)
        actor_logs = self.actor.update(train_data, self.critic)
        
        if self.num_param_updates % self.target_update_freq == 0:
            self.critic.update_target_network()
        
        self.num_param_updates += 1
        
        return {**critic_logs, **actor_logs}
