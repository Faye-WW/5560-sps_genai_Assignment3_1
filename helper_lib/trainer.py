from __future__ import annotations
from typing import Optional, Dict, Any
import time
import torch
import torch.nn as nn
from torch.optim import Optimizer

from .evaluator import evaluate
from .utils import save_checkpoint


def train_model(
    model: nn.Module,
    train_loader,
    val_loader=None,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    device: str | torch.device = "cpu",
    epochs: int = 5,
    log_interval: int = 50,
    save_path: Optional[str] = "models/cnn.pt",
) -> Dict[str, Any]:
    """
    Generic classification training loop.

    Returns a dict with best metrics:
      {
        "best_acc": float,
        "best_epoch": int,
        "last_train_loss": float,
        "last_val_acc": float,
      }
    """
    device = torch.device(device)
    model.to(device)
    model.train()

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc = -1.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        seen = 0

        for i, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_loss += loss.item() * bs
            seen += bs

            if i % log_interval == 0:
                print(f"[Epoch {epoch:03d} | {i:04d}/{len(train_loader)}] "
                      f"loss={running_loss/seen:.4f}")

        train_loss = running_loss / max(seen, 1)
        msg = f"Epoch {epoch:03d} finished: train_loss={train_loss:.4f}"

        val_acc = None
        if val_loader is not None:
            eval_res = evaluate(model, val_loader, criterion, device=device)
            val_acc = eval_res["accuracy"]
            msg += f"  val_acc={val_acc:.4f}"

            # Save best
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                if save_path:
                    save_checkpoint(model, save_path)
        else:
            if save_path:
                save_checkpoint(model, save_path)

        dt = time.time() - t0
        print(msg + f"  ({dt:.1f}s)")

    return {
        "best_acc": float(best_acc),
        "best_epoch": int(best_epoch),
        "last_train_loss": float(train_loss),
        "last_val_acc": float(val_acc) if val_loader is not None else None,
        "saved_to": save_path,
    }
