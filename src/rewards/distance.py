import torch


def compute_distance(distance_type: str, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    if distance_type == 'l1':
        distance = torch.abs(goal - state)
    elif distance_type == 'l2':
        distance = torch.sqrt(torch.pow(goal - state, exponent=2))
    elif distance_type == 'cosine':
        distance = torch.nn.functional.cosine_similarity(state, goal, dim=-1)
    else:
        raise ValueError(f"Unknown goal norm type '{distance_type}'")
    
    return distance
