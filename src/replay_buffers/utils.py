from omegaconf import DictConfig
from tensordict import TensorDict


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
