import torch
from tensordict import TensorDict


def get_tensor(tensordict: TensorDict, keys: list) -> torch.Tensor:
    return torch.cat([tensordict.get(in_key, None) for in_key in keys], dim=-1)
