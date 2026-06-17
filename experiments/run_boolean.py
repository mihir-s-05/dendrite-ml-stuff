"""Matched-compute comparison of MLP / SwiGLU / Dendritic FFN blocks on the
Boolean-function suite from "What can a neuron compute?" (parity + random
balanced functions).

Each block is sized to the SAME parameter budget so any accuracy gap reflects
inductive bias, not capacity. A single FFN block (n_layers=1) is used to
mirror the paper's single-unit framing.

Usage:
    python experiments/run_boolean.py
    python experiments/run_boolean.py --d-list 2 4 6 8 10 --target-params 20000 --seeds 3
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from src.blocks import MLP, SwiGLU, DendriticFFN, BlockClassifier
from src.counting import count_params, size_to_budget
from src.tasks import parity_fn, random_balanced_boolean, make_dataset
from src.train import set_seed, train_classifier, pick_device


def build_factory(name: str, d_model: int, target_params: int):
    """Return (make_block, hidden, block_params) sized to the budget."""
    if name == "mlp":
        size = lambda h: MLP(d_model, h)
        h = size_to_budget(size, target_params)
        return (lambda: MLP(d_model, h)), h
    if name == "swiglu":
        size = lambda h: SwiGLU(d_model, h)
        h = size_to_budget(size, target_params)
        return (lambda: SwiGLU(d_model, h)), h
    if name == "dendritic":
        # Keep ~8 units per branch; grow the number of branches with the budget.
        bdim = 8
        size = lambda K: DendriticFFN(d_model, n_branches=max(1, K), branch_dim=bdim)
        K = size_to_budget(size, target_params, lo=1, hi=2048)
        return (lambda: DendriticFFN(d_model, n_branches=max(1, K), branch_dim=bdim)), K
    if name == "dendritic_local":
        # Faithful version of the paper's prior: each branch reads only its own
        # slice of the input (local sub-units). Locality frees input-side
        # params, so at matched budget this gets more branches/soma width.
        bdim = 8
        size = lambda K: DendriticFFN(d_model, n_branches=max(1, K), branch_dim=bdim,
                                      local_input=True)
        K = size_to_budget(size, target_params, lo=1, hi=4096)
        return (lambda: DendriticFFN(d_model, n_branches=max(1, K), branch_dim=bdim,
                                     local_input=True)), K
    raise ValueError(name)


MODELS = ["mlp", "swiglu", "dendritic", "dendritic_local"]


def run_one(model_name, d, fn, args, seed):
    set_seed(seed)
    Xtr, ytr, Xte, yte = make_dataset(
        d, fn, n_per_pattern=args.samples, noise_std=args.noise, seed=seed
    )
    make_block, _ = build_factory(model_name, args.d_model, args.target_params)
    model = BlockClassifier(d_in=d, d_model=args.d_model, make_block=make_block, n_layers=1)
    res = train_classifier(
        model, Xtr, ytr, Xte, yte,
        epochs=args.epochs, lr=args.lr, device=args.device,
    )
    return res["best_test_acc"], count_params(make_block())


# CPU = quick PoC; GPU = harder/heavier sweep for the 4070 (up to 10-bit
# parity, bigger budget, more seeds). CLI flags override preset values.
PRESETS = {
    "cpu": dict(d_list=[2, 4, 6, 8], target_params=20000, d_model=64,
                epochs=300, lr=3e-3, noise=0.5, samples=64, seeds=3,
                n_random_rules=8),
    "gpu": dict(d_list=[2, 4, 6, 8, 10], target_params=60000, d_model=128,
                epochs=800, lr=3e-3, noise=0.5, samples=128, seeds=5,
                n_random_rules=16),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="cpu")
    ap.add_argument("--d-list", type=int, nargs="+", default=None)
    ap.add_argument("--target-params", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--n-random-rules", type=int, default=None)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    for key, val in cfg.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    args.device = pick_device(args.device)

    print(f"\nDevice: {args.device} (preset={args.preset}). Budget ~{args.target_params} "
          f"params/block, d_model={args.d_model}, {args.seeds} seeds, single FFN block.\n")
    for m in MODELS:
        _, h = build_factory(m, args.d_model, args.target_params)
        p = count_params(build_factory(m, args.d_model, args.target_params)[0]())
        tag = "branches" if m.startswith("dendritic") else "hidden"
        print(f"  {m:10s}  {tag}={h:<5d} actual_params={p}")

    print("\n=== PARITY (test accuracy %, mean +/- sd) ===")
    header = "  d   " + "".join(f"{m:>16s}" for m in MODELS)
    print(header)
    for d in args.d_list:
        fn = parity_fn(d)
        row = f"  {d:<3d} "
        for m in MODELS:
            accs = [run_one(m, d, fn, args, s) [0] for s in range(args.seeds)]
            row += f"{np.mean(accs)*100:8.1f}+-{np.std(accs)*100:4.1f}"
        print(row)

    print("\n=== RANDOM BALANCED 4-bit BOOLEAN (test accuracy %, mean +/- sd) ===")
    d = 4
    row = "  4   "
    for m in MODELS:
        accs = []
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 + s)
            for r in range(args.n_random_rules):
                fn = random_balanced_boolean(d, rng)
                accs.append(run_one(m, d, fn, args, seed=2000 + s * 100 + r)[0])
        row += f"{np.mean(accs)*100:8.1f}+-{np.std(accs)*100:4.1f}"
    print(row)
    print()


if __name__ == "__main__":
    main()
