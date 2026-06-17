"""Permuted-MNIST continual learning: the standard Active-Dendrites benchmark
(Iyer et al. 2022). Each task applies a fixed random pixel permutation to all
images; tasks are learned sequentially with NO replay, given a one-hot task
context. We measure retention across tasks.

Models:
  - mlp            : ignores context (control; forgets).
  - concat         : context concatenated to input.
  - kwta_only      : sparse (kWTA) MLP, ignores context (isolates sparsity).
  - active_dendrite: context-gated dendrites + kWTA (the method).

Metrics: final mean test accuracy over all tasks, and forgetting (acc right
after learning a task minus acc at the end).

Usage:
    uv run --no-sync python -u experiments/run_pmnist.py --device cuda
    uv run --no-sync python -u experiments/run_pmnist.py --device cuda --preset big
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from src.blocks import ContextMLP, ConcatContextMLP, ActiveDendriteMLP, KWTAMLP
from src.counting import count_params
from src.mnist import load_mnist
from src.train import set_seed, pick_device

MODELS = {
    "mlp": ContextMLP,
    "concat": ConcatContextMLP,
    "kwta_only": KWTAMLP,
    "active_dendrite": ActiveDendriteMLP,
}


def make_permutations(n_tasks, dim, seed):
    rng = np.random.default_rng(seed)
    return [torch.from_numpy(rng.permutation(dim)) for _ in range(n_tasks)]


def onehot(t, n, rows, device):
    v = torch.zeros(rows, n, device=device)
    v[:, t] = 1.0
    return v


@torch.no_grad()
def eval_task(model, X, y, perm, t, n_tasks, device, bs=2000):
    model.eval()
    correct = 0
    for i in range(0, len(X), bs):
        xb = X[i : i + bs][:, perm].to(device)
        yb = y[i : i + bs].to(device)
        ctx = onehot(t, n_tasks, len(xb), device)
        correct += (model(xb, ctx).argmax(-1) == yb).sum().item()
    return correct / len(X)


def train_task(model, X, y, perm, t, n_tasks, epochs, lr, bs, device):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    n = len(X)
    for _ in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i : i + bs]
            xb = X[b][:, perm].to(device)
            yb = y[b].to(device)
            ctx = onehot(t, n_tasks, len(xb), device)
            opt.zero_grad()
            loss_fn(model(xb, ctx), yb).backward()
            opt.step()


def run_model(name, Xtr, ytr, Xte, yte, perms, args, device):
    set_seed(args.seed)
    model = MODELS[name](d_in=784, n_ctx=len(perms), d_hidden=args.hidden,
                         n_out=10, n_segments=args.segments,
                         kwta_frac=args.kwta_frac, dend_init=args.dend_init).to(device)
    acc_after = []
    for t, perm in enumerate(perms):
        train_task(model, Xtr, ytr, perm, t, len(perms), args.epochs, args.lr,
                   args.batch, device)
        acc_after.append(eval_task(model, Xte, yte, perm, t, len(perms), device))
    acc_final = [eval_task(model, Xte, yte, perm, t, len(perms), device)
                 for t, perm in enumerate(perms)]
    forgetting = float(np.mean([acc_after[t] - acc_final[t]
                                for t in range(len(perms) - 1)]))
    return count_params(model), float(np.mean(acc_final)), forgetting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["fast", "big"], default="fast")
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--segments", type=int, default=10)
    ap.add_argument("--kwta-frac", type=float, default=0.1)
    ap.add_argument("--dend-init", type=float, default=1.5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()
    if args.preset == "big":
        args.hidden, args.epochs, args.train_n, args.n_tasks = 2048, 5, 60000, 10
    args.device = pick_device(args.device)

    Xtr, ytr, Xte, yte = load_mnist()
    Xtr, ytr = Xtr[: args.train_n], ytr[: args.train_n]
    perms = make_permutations(args.n_tasks, 784, args.seed)

    print(f"\nDevice: {args.device} (preset={args.preset}). Permuted-MNIST, "
          f"{args.n_tasks} tasks, hidden={args.hidden}, segments={args.segments}, "
          f"kwta={args.kwta_frac}, {args.epochs} epochs/task, train_n={args.train_n}, "
          f"sequential/no-replay.\n")
    print(f"{'model':16s}{'params':>11s}{'final_mean_acc':>16s}{'forgetting':>13s}")
    for name in MODELS:
        p, acc, forg = run_model(name, Xtr, ytr, Xte, yte, perms, args, args.device)
        print(f"{name:16s}{p:>11d}{acc*100:>15.1f}%{forg*100:>12.1f}%", flush=True)
    print()


if __name__ == "__main__":
    main()
