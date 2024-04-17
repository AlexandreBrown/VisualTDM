import torch


def compute_distance(norm_type: str, obs_latent: torch.Tensor, goal_latent: torch.Tensor) -> torch.Tensor:
    if norm_type == 'l1':
        distance = torch.abs(goal_latent - obs_latent)
    elif norm_type == 'l2':
        distance = torch.sqrt(torch.pow(goal_latent - obs_latent, exponent=2))
    else:
        raise ValueError(f"Unknown goal norm type '{norm_type}'")
    
    return distance
