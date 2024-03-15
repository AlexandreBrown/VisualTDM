import h5py
import torch
from torch.utils.data import Dataset
from pathlib import Path


class VAEDataset(Dataset):
    def __init__(self, data_path: Path, transform = None):
        self.data_path = data_path
        self.dataset_name = data_path.stem
        self.transform = transform
        self.dataset_file = h5py.File(data_path, "r")
        self.dataset = self.dataset_file[self.dataset_name]
        self.dataset_size = self.dataset.shape[0]
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        images_hwc = self.dataset[idx]

        images_chw = torch.from_numpy(images_hwc).permute(2, 0, 1)
        
        if self.transform:
            images_chw = self.transform(images_chw)
        
        return images_chw
    
    def close(self):
        self.dataset_file.close()
