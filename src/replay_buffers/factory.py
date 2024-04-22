from omegaconf import DictConfig
from torchrl.data.replay_buffers import LazyMemmapStorage
from torchrl.data import TensorDictReplayBuffer
from torchrl.data import SliceSamplerWithoutReplacement
from torchrl.data.replay_buffers import ReplayBuffer


def create_replay_buffer(cfg: DictConfig) -> ReplayBuffer:
    return TensorDictReplayBuffer(storage=LazyMemmapStorage(max_size=cfg['replay_buffer']['max_size']),
                                  sampler=SliceSamplerWithoutReplacement(traj_key="traj", num_slices=cfg['train']['num_trajs'], strict_length=False))
