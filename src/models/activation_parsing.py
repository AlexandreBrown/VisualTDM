import torch.nn.functional as F
from functools import partial
from typing import Union
import torch.nn as nn

def get_activation(name: str, leaky_relu_negative_slope:Union[float]=None):
    if name == "relu":
        activation_fn = F.relu
    elif name == "leaky_relu":
        activation_fn = F.leaky_relu
        activation_fn = partial(activation_fn, negative_slope=leaky_relu_negative_slope)
    elif name == "tanh":
        activation_fn = F.tanh
    elif name == "identity":
        activation_fn = nn.Identity()
    else:
        raise ValueError(f"Unknown encoder activation function '{name}'")
    
    return activation_fn
