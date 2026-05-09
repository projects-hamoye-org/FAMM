from pathlib import Path
import tifffile
import torch
from torch.utils.data import Dataset
import numpy as np

class MultispectralTiffDataset(Dataset):
    def __init__(self, root_dir, num_bands=10, transform=None):
        self.root_dir = Path(root_dir)
        self.num_bands = num_bands
        self.transform = transform

        self.samples = []
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = int(class_dir.name.split("_")[0])  # 0 or 1
            for tif_path in class_dir.glob("*.tif"):
                self.samples.append((tif_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = tifffile.imread(path).astype(np.float32)
        img = np.nan_to_num(img, nan=0.0)

        if img.ndim == 2:
            img = img[..., None]

        img = img[:, :, :self.num_bands]  # keep 10 bands
        img = torch.from_numpy(img).permute(2, 0, 1)  # [C,H,W]

        if self.transform:
            img = self.transform(img)

        return img, label