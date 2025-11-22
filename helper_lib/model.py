# helper_lib/model.py
# Models and a simple factory to retrieve them by name.

from __future__ import annotations
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


class AssignmentCNN(nn.Module):
    """
    CNN architecture required by the assignment:
    Input: RGB 64x64
    [Conv(3->16, k3,s1,p1) + ReLU + MaxPool(2,2)]
    [Conv(16->32, k3,s1,p1) + ReLU + MaxPool(2,2)]
    Flatten -> FC(8192->100) + ReLU -> FC(100->num_classes)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Flatten dim: 32 * 16 * 16 = 8192 (for 64x64 input)
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, 64, 64)
        x = self.pool1(F.relu(self.conv1(x)))  # (N, 16, 32, 32)
        x = self.pool2(F.relu(self.conv2(x)))  # (N, 32, 16, 16)
        x = torch.flatten(x, 1)                # (N, 8192)
        x = F.relu(self.fc1(x))                # (N, 100)
        x = self.fc2(x)                        # (N, num_classes)
        return x


def get_model(
    name: Literal["CNN", "AssignmentCNN", "VAE", "EnhancedCNN"] = "CNN",
    num_classes: int = 10,
) -> nn.Module:
    """
    Simple factory to construct models by name.

    Parameters
    ----------
    name : str
        "CNN" / "AssignmentCNN" -> AssignmentCNN (classification)
        "VAE" / "EnhancedCNN"   -> placeholders (raise for now)
    num_classes : int
        Number of output classes (default 10)

    Returns
    -------
    torch.nn.Module
    """
    key = name.lower()
    if key in ("cnn", "assignmentcnn"):
        return AssignmentCNN(num_classes=num_classes)

    # Placeholders for future modules (you'll implement later)
    if key == "vae":
        raise NotImplementedError("VAE will be added in Module 5.")
    if key == "enhancedcnn":
        raise NotImplementedError("EnhancedCNN not implemented yet.")

    raise ValueError(f"Unknown model name: {name}")
