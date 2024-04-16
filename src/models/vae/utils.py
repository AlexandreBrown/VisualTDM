import torch
from tensordict.nn import TensorDictModule
from tensordict import TensorDict


def encode_to_latent_representation(encoder: TensorDictModule,
                                     image: torch.Tensor,
                                     device: torch.device) -> torch.Tensor:  
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
        batch_size = 1
        squeeze_output = True
    else:
        batch_size = image.shape[0]
        squeeze_output = False
    
    input = TensorDict(
        source={
            "image": image.to(device)
        },
        batch_size=batch_size
    )
    
    output = encoder.to(device)(input)['q_z'].loc
    if squeeze_output:
        output = output.squeeze(0)
    return output
