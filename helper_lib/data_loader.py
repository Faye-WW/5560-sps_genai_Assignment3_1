"""
Data loading utilities for 64x64 RGB images.

- get_cifar10_loaders: quick start with CIFAR-10 (10 classes)
- get_imagefolder_loaders: generic loader for your own dataset laid out as:
    dataset/
      train/classA/xxx.jpg
      train/classB/yyy.jpg
      val/classA/zzz.jpg
      ...
"""

from typing import Tuple, Optional
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as D

def _default_transforms(img_size: int = 64) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        # Good generic normalization for RGB; adjust if you know dataset stats
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

def get_cifar10_loaders(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    img_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    tfm = _default_transforms(img_size)
    trainset = D.CIFAR10(root=root, train=True, download=True, transform=tfm)
    testset  = D.CIFAR10(root=root, train=False, download=True, transform=tfm)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

def get_imagefolder_loaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 2,
    img_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    """
    root should contain 'train' and 'val' subfolders.
    """
    tfm = _default_transforms(img_size)
    trainset = D.ImageFolder(f"{root}/train", transform=tfm)
    valset   = D.ImageFolder(f"{root}/val",   transform=tfm)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(valset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
