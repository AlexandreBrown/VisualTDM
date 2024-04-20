import torch


def compute_distance(distance_type: str, obs_latent: torch.Tensor, goal_latent: torch.Tensor) -> torch.Tensor:
    if distance_type == 'l1':
        distance = torch.abs(goal_latent - obs_latent)
    elif distance_type == 'l2':
        distance = torch.sqrt(torch.pow(goal_latent - obs_latent, exponent=2))
    elif distance_type == 'cosine':
        distance = torch.nn.functional.cosine_similarity(obs_latent, goal_latent, dim=-1)
    else:
        raise ValueError(f"Unknown goal norm type '{distance_type}'")
    
    return distance
