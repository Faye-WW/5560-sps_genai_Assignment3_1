from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader,
    criterion: nn.Module | None = None,
    device: str | torch.device = "cpu",
) -> Dict[str, float]:
    """
    Evaluate classification model: returns {"loss": avg_loss, "accuracy": acc}
    """
    device = torch.device(device)
    model.eval()
    model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total += bs
        total_loss += loss.item() * bs
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    return {"loss": float(avg_loss), "accuracy": float(acc)}
