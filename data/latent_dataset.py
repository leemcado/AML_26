# data/latent_dataset.py — VAE latent dataset
# Loads pre-extracted .pt tensors from disk (bypasses VAE encoder)

import os
import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    """Pre-extracted VAE latent tensor dataset.

    Directory structure:
        latent_dir/
        ├── 0/          # class 0
        │   ├── 0000.pt
        │   └── ...
        └── ...

    Each .pt: {"latent": (4,32,32), "label": int}
    """

    def __init__(self, latent_dir, transform=None):
        super().__init__()
        self.latent_dir = latent_dir
        self.transform = transform
        self.samples = []

        for class_name in sorted(os.listdir(latent_dir)):
            class_dir = os.path.join(latent_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            try:
                class_idx = int(class_name)
            except ValueError:
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.endswith('.pt'):
                    fpath = os.path.join(class_dir, fname)
                    self.samples.append((fpath, class_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No .pt files found in {latent_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        data = torch.load(fpath, map_location='cpu', weights_only=True)

        if isinstance(data, dict):
            latent = data['latent']
        else:
            latent = data

        latent = latent.float()

        if self.transform is not None:
            latent = self.transform(latent)

        return latent, label


class DummyLatentDataset(Dataset):
    """Random latent dataset for debugging."""

    def __init__(self, num_samples=1000, num_classes=1000,
                 latent_shape=(4, 32, 32)):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.latent_shape = latent_shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        latent = torch.randn(self.latent_shape)
        label = idx % self.num_classes
        return latent, label
