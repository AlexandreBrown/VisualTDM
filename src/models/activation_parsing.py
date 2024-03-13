import torch.nn.functional as F
from functools import partial
from typing import Union

def get_activation(activation_type: str, leaky_relu_negative_slope:Union[float]=None):
    if activation_type == "relu":
        activation_fn = F.relu
    elif activation_type == "leaky_relu":
        activation_fn = F.leaky_relu
        activation_fn = partial(activation_fn, negative_slope=leaky_relu_negative_slope)
    else:
        raise ValueError(f"Unknown encoder activation function '{activation_type}'")
    
    return activation_fn
