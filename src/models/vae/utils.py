import torch
import torch.nn.functional as F
from tensordict.nn import TensorDictModule
from tensordict import TensorDict

from models.vae.model import VAEDecoder


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

def decode_to_rgb(decoder: VAEDecoder,
                  latent: torch.Tensor) -> torch.Tensor:
    decoded_x = decoder(latent).loc.squeeze(0).cpu()
    decoded_x = F.sigmoid(decoded_x)
    decoded_x = torch.clamp(decoded_x * 255, min=0, max=255).to(torch.uint8)
    
    return decoded_x
    