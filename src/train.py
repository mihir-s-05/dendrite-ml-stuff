"""Minimal training / evaluation loop for the block classifiers."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(choice: str = "auto") -> str:
    """Resolve a device string; "auto" prefers CUDA when present."""
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    logits = model(X).squeeze(-1)
    pred = (logits > 0).float()
    return (pred == y).float().mean().item()


def split_param_groups(model: nn.Module, weight_decay: float) -> list:
    """Decay only 2D+ weight matrices; never decay biases, norm gains, or the
    dendritic routing/segment params.

    Decaying `route_logits` (init 0) toward 0 pins the routing softmax at
    uniform, which artificially prevents context specialization. Excluding it
    (and other structural params) follows the standard transformer recipe and
    removes that confound.
    """
    no_decay_names = ("route_logits", "dend", "base_times")
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or any(k in name for k in no_decay_names):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train_classifier(
    model: nn.Module,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xte: torch.Tensor,
    yte: torch.Tensor,
    epochs: int = 300,
    lr: float = 3e-3,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    verbose: bool = False,
    warmup_frac: float = 0.05,
    grad_clip: float = 1.0,
) -> dict:
    model.to(device)
    Xtr, ytr, Xte, yte = (t.to(device) for t in (Xtr, ytr, Xte, yte))
    opt = torch.optim.AdamW(split_param_groups(model, weight_decay), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(Xtr)
    best_test = 0.0

    steps_per_epoch = max(1, math.ceil(n / batch_size))
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(warmup_frac * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            logits = model(Xtr[idx]).squeeze(-1)
            loss = loss_fn(logits, ytr[idx])
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            te = accuracy(model, Xte, yte)
            best_test = max(best_test, te)
            if verbose:
                tr = accuracy(model, Xtr, ytr)
                print(f"  epoch {ep+1:4d}  train {tr:.3f}  test {te:.3f}")

    return {"test_acc": accuracy(model, Xte, yte), "best_test_acc": best_test}
