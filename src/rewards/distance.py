import torch


def compute_distance(distance_type: str, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    if distance_type == 'l1':
        distance = torch.abs(goal - state)
    elif distance_type == 'squared_diff':
        distance = (goal - state)**2
    elif distance_type == 'cosine':
        distance = torch.nn.functional.cosine_similarity(state, goal, dim=-1)
    else:
        raise ValueError(f"Unknown goal norm type '{distance_type}'")
    
    return distance
