from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn


def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def save_checkpoint(model: nn.Module, path: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(model.state_dict(), path)
    print(f"[utils] Saved checkpoint to: {path}")


def load_checkpoint(model: nn.Module, path: str, map_location="cpu") -> nn.Module:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model
