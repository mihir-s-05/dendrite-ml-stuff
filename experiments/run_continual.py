"""Continual / multi-task learning test (the angle that has actually worked for
dendrites in ML: Iyer et al. 2022, "Avoiding Catastrophe: Active Dendrites
Enable Multiple Task Learning in Dynamic Environments").

Setup: a sequence of T conflicting tasks over the SAME input space (each task
is a different random balanced Boolean function over d bits, so labels collide
across tasks). The model is trained sequentially, one task at a time, with NO
replay, and is given a one-hot task/context vector. We then measure how much it
remembers earlier tasks.

Models:
  - mlp            : ignores context (control; should forget).
  - concat         : context concatenated to the input (naive use of context).
  - active_dendrite: context-gated dendritic units (Iyer et al. 2022).

Metrics (higher acc / lower forgetting is better):
  - final mean acc : mean test accuracy over all tasks after the full sequence.
  - forgetting     : mean over earlier tasks of (acc right after learning it
                     minus acc at the end).

Usage:
    uv run --no-sync python -u experiments/run_continual.py
    uv run --no-sync python -u experiments/run_continual.py --preset gpu
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from src.blocks import ContextMLP, ConcatContextMLP, ActiveDendriteMLP, KWTAMLP
from src.counting import count_params
from src.tasks import random_balanced_boolean, make_dataset
from src.train import set_seed, pick_device

MODELS = {
    "mlp": ContextMLP,
    "concat": ConcatContextMLP,
    "kwta_only": KWTAMLP,
    "active_dendrite": ActiveDendriteMLP,
}


def build_tasks(n_tasks, d, samples, noise, seed):
    rng = np.random.default_rng(seed)
    tasks = []
    for t in range(n_tasks):
        fn = random_balanced_boolean(d, rng)
        data = make_dataset(d, fn, n_per_pattern=samples, noise_std=noise,
                            seed=seed + 1 + t)
        tasks.append(data)
    return tasks


def onehot(t, n, n_rows, device):
    v = torch.zeros(n_rows, n, device=device)
    v[:, t] = 1.0
    return v


@torch.no_grad()
def eval_task(model, X, y, t, n_tasks, device):
    model.eval()
    X, y = X.to(device), y.to(device)
    ctx = onehot(t, n_tasks, len(X), device)
    pred = (model(X, ctx).squeeze(-1) > 0).float()
    return (pred == y).float().mean().item()


def train_task(model, X, y, t, n_tasks, epochs, lr, batch, device):
    model.train()
    X, y = X.to(device), y.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            ctx = onehot(t, n_tasks, len(idx), device)
            opt.zero_grad()
            loss = loss_fn(model(X[idx], ctx).squeeze(-1), y[idx])
            loss.backward()
            opt.step()


def run_model(name, tasks, args, device):
    set_seed(args.seed)
    Cls = MODELS[name]
    model = Cls(d_in=args.d, n_ctx=len(tasks), d_hidden=args.hidden,
                n_out=1, n_segments=args.segments, kwta_frac=args.kwta_frac).to(device)
    acc_after = []  # accuracy on task t right after training it
    for t, (Xtr, ytr, Xte, yte) in enumerate(tasks):
        train_task(model, Xtr, ytr, t, len(tasks), args.epochs, args.lr,
                   args.batch, device)
        acc_after.append(eval_task(model, Xte, yte, t, len(tasks), device))
    acc_final = [eval_task(model, te[2], te[3], t, len(tasks), device)
                 for t, te in enumerate(tasks)]
    forgetting = float(np.mean([acc_after[t] - acc_final[t]
                                for t in range(len(tasks) - 1)]))
    return {
        "params": count_params(model),
        "final_mean": float(np.mean(acc_final)),
        "forgetting": forgetting,
        "acc_final": acc_final,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--n-tasks", type=int, default=5)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--kwta-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()
    if args.preset == "gpu":
        args.hidden, args.epochs, args.samples, args.n_tasks = 512, 120, 128, 8
    args.device = pick_device(args.device)

    print(f"\nDevice: {args.device} (preset={args.preset}).  "
          f"{args.n_tasks} conflicting {args.d}-bit tasks, hidden={args.hidden}, "
          f"sequential, no replay.\n")

    tasks = build_tasks(args.n_tasks, args.d, args.samples, args.noise, args.seed)
    print(f"{'model':16s}{'params':>10s}{'final_mean_acc':>16s}{'forgetting':>13s}")
    for name in MODELS:
        r = run_model(name, tasks, args, args.device)
        print(f"{name:16s}{r['params']:>10d}{r['final_mean']*100:>15.1f}%"
              f"{r['forgetting']*100:>12.1f}%")
    print()


if __name__ == "__main__":
    main()
