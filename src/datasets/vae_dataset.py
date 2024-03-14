import h5py
import torch
from torch.utils.data import Dataset
from pathlib import Path


class VAEDataset(Dataset):
    def __init__(self, data_path: Path, dataset_name: str, transform):
        self.data_path = data_path
        self.transform = transform
        with h5py.File(data_path, "r") as file:
            self.dataset_size = file[dataset_name].shape[0]
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        with h5py.File(self.data_path, 'r') as file:
            images_hwc = file['images'][idx]

        images_chw = torch.from_numpy(images_hwc).permute(0, 3, 1, 2)
        
        if self.transform:
            images_chw = self.transform(images_chw)
        
        return images_chw
