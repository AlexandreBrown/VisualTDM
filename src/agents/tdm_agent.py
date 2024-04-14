import torch
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from actors.tdm_actor import TdmActor
from critics.tdm_critic import TdmCritic

class TdmAgent():
    def __init__(self, 
                 target_update_freq: int,
                 obs_dim: int,
                 actions_dim: int,
                 goal_latent_dim: int,
                 fc1_out_features: int,
                 device: torch.device,
                 norm_type: str,
                 encoder: TensorDictModule,
                 critic_learning_rate: float,
                 actor_learning_rate: float,
                 polyak_avg: float):
        self.actor = TdmActor(obs_dim=obs_dim,
                              actions_dim=actions_dim,
                              goal_latent_dim=goal_latent_dim,
                              fc1_out_features=fc1_out_features,
                              device=device,
                              learning_rate=actor_learning_rate,
                              polyak_avg=polyak_avg)
        self.critic = TdmCritic(obs_dim=obs_dim,
                                actions_dim=actions_dim,
                                goal_latent_dim=goal_latent_dim,
                                fc1_out_features=fc1_out_features,
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
        logs = {}
        train_data = train_data.to(self.device)
        self.critic.update(train_data, logs)
        self.actor.update(train_data, self.critic, logs)
        
        if self.num_param_updates % self.target_update_freq == 0:
            self.critic.update_target_network()
        
        self.num_param_updates += 1
        
        return logs
