"""Spatiotemporal Boolean benchmark for transformer-style dendritic FFNs.

This is the paper-faithful follow-up to `run_boolean.py`: instead of static
noisy +/-1 vectors, each Boolean pattern is encoded as timed ON/OFF afferent
spikes with jitter. Models are sequence classifiers built from transformer FFN
sub-layers, and every FFN block is sized to a matched parameter budget.

Usage:
    uv run --no-sync python -u experiments/run_temporal_boolean.py --preset smoke
    uv run --no-sync python -u experiments/run_temporal_boolean.py --preset gpu --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from src.blocks import (
    MLP,
    SwiGLU,
    DendriticFFN,
    TemporalDendriticFFN,
    TemporalBlockClassifier,
    CausalConvSwiGLU,
    GatedConvFFN,
)
from src.counting import count_params, size_to_budget
from src.tasks import parity_fn, random_balanced_boolean, make_temporal_boolean_dataset
from src.train import pick_device, set_seed, train_classifier


MODELS = [
    "mlp",
    "swiglu",
    "temporal_conv",
    "gated_conv",
    "dendritic",
    "temporal_dendritic",
    "temporal_no_route",
    "temporal_no_memory",
    "temporal_no_nmda",
    "temporal_no_interaction",
]


def build_factory(name: str, d_model: int, target_params: int, memory_kernel: int):
    """Return (make_block, size_tag) sized to the block budget."""
    if name == "mlp":
        h = size_to_budget(lambda h: MLP(d_model, h), target_params)
        return (lambda: MLP(d_model, h)), h
    if name == "swiglu":
        h = size_to_budget(lambda h: SwiGLU(d_model, h), target_params)
        return (lambda: SwiGLU(d_model, h)), h
    if name == "temporal_conv":
        h = size_to_budget(
            lambda h: CausalConvSwiGLU(d_model, h, memory_kernel=memory_kernel), target_params)
        return (lambda: CausalConvSwiGLU(d_model, h, memory_kernel=memory_kernel)), h
    if name == "gated_conv":
        h = size_to_budget(
            lambda h: GatedConvFFN(d_model, h, memory_kernel=memory_kernel), target_params)
        return (lambda: GatedConvFFN(d_model, h, memory_kernel=memory_kernel)), h
    if name == "dendritic":
        bdim = 8
        size = lambda K: DendriticFFN(d_model, n_branches=max(1, K), branch_dim=bdim)
        K = size_to_budget(size, target_params, lo=1, hi=2048)
        return (lambda: DendriticFFN(d_model, n_branches=max(1, K), branch_dim=bdim)), K

    bdim = 8
    kwargs = {
        "memory_kernel": memory_kernel,
        "nmda_gate": True,
        "branch_interaction": True,
        "routed_input": True,
    }
    if name == "temporal_no_route":
        kwargs["routed_input"] = False
    elif name == "temporal_no_memory":
        kwargs["memory_kernel"] = 1
    elif name == "temporal_no_nmda":
        kwargs["nmda_gate"] = False
    elif name == "temporal_no_interaction":
        kwargs["branch_interaction"] = False
    elif name != "temporal_dendritic":
        raise ValueError(name)

    size = lambda K: TemporalDendriticFFN(
        d_model,
        n_branches=max(1, K),
        branch_dim=bdim,
        **kwargs,
    )
    K = size_to_budget(size, target_params, lo=1, hi=2048)
    return (
        lambda: TemporalDendriticFFN(d_model, n_branches=max(1, K), branch_dim=bdim, **kwargs),
        K,
    )


def temporal_modules(model: torch.nn.Module) -> list[TemporalDendriticFFN]:
    return [m for m in model.modules() if isinstance(m, TemporalDendriticFFN)]


@torch.no_grad()
def routing_entropy(model: torch.nn.Module) -> float | None:
    mods = temporal_modules(model)
    routed = [m.routing_entropy().item() for m in mods if m.routed_input]
    if not routed:
        return None
    return float(np.mean(routed))


def make_data(args, d, fn, seed):
    return make_temporal_boolean_dataset(
        d,
        fn,
        n_per_pattern=args.samples,
        n_time=args.n_time,
        axons_per_bit=args.axons_per_bit,
        jitter_std=args.jitter,
        spike_width=args.spike_width,
        background_rate=args.background_rate,
        seed=seed,
    )


def run_one(model_name, data, args, seed):
    """Train one model on a PRE-BUILT dataset (built once per (d, seed) and
    reused across all models, so data generation isn't redone 9x)."""
    Xtr, ytr, Xte, yte = data
    set_seed(seed)
    make_block, _ = build_factory(
        model_name,
        args.d_model,
        args.target_params,
        args.memory_kernel,
    )
    model = TemporalBlockClassifier(
        d_in=Xtr.shape[-1],
        d_model=args.d_model,
        make_block=make_block,
        n_layers=args.n_layers,
        decision_window=args.decision_window,
    )
    res = train_classifier(
        model,
        Xtr,
        ytr,
        Xte,
        yte,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )
    return res["best_test_acc"], count_params(make_block()), routing_entropy(model)


PRESETS = {
    "smoke": dict(
        d_list=[2, 4],
        target_params=20000,
        d_model=64,
        n_layers=1,
        epochs=40,
        lr=3e-3,
        batch_size=256,
        samples=16,
        n_time=32,
        axons_per_bit=4,
        decision_window=8,
        memory_kernel=7,
        jitter=1.25,
        spike_width=0.5,
        background_rate=0.001,
        seeds=1,
        n_random_rules=2,
    ),
    "cpu": dict(
        d_list=[2, 4, 6],
        target_params=30000,
        d_model=96,
        n_layers=1,
        epochs=120,
        lr=3e-3,
        batch_size=256,
        samples=32,
        n_time=48,
        axons_per_bit=6,
        decision_window=10,
        memory_kernel=9,
        jitter=1.5,
        spike_width=0.75,
        background_rate=0.002,
        seeds=2,
        n_random_rules=4,
    ),
    "gpu": dict(
        d_list=[2, 4, 6, 8, 10],
        target_params=80000,
        d_model=128,
        n_layers=1,
        epochs=250,
        lr=2e-3,
        batch_size=512,
        samples=48,
        n_time=64,
        axons_per_bit=8,
        decision_window=12,
        memory_kernel=11,
        jitter=1.5,
        spike_width=0.75,
        background_rate=0.002,
        seeds=3,
        n_random_rules=8,
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="smoke")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--d-list", type=int, nargs="+", default=None)
    ap.add_argument("--target-params", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--n-time", type=int, default=None)
    ap.add_argument("--axons-per-bit", type=int, default=None)
    ap.add_argument("--decision-window", type=int, default=None)
    ap.add_argument("--memory-kernel", type=int, default=None)
    ap.add_argument("--jitter", type=float, default=None)
    ap.add_argument("--spike-width", type=float, default=None)
    ap.add_argument("--background-rate", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--n-random-rules", type=int, default=None)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    for key, val in cfg.items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    args.device = pick_device(args.device)

    print(
        f"\nDevice: {args.device} (preset={args.preset}). Spatiotemporal Boolean, "
        f"T={args.n_time}, axons/bit={args.axons_per_bit}, jitter={args.jitter}, "
        f"budget~{args.target_params} params/block.\n"
    )
    print("Block sizes:")
    for m in args.models:
        make_block, size = build_factory(m, args.d_model, args.target_params, args.memory_kernel)
        tag = "hidden" if m in ("mlp", "swiglu", "temporal_conv", "gated_conv") else "branches"
        print(f"  {m:24s} {tag}={size:<5d} actual_params={count_params(make_block())}")

    print("\n=== TEMPORAL PARITY (test accuracy %, mean +/- sd) ===")
    header = "  d   " + "".join(f"{m:>24s}" for m in args.models)
    print(header)
    for d in args.d_list:
        fn = parity_fn(d)
        accs = {m: [] for m in args.models}
        for s in range(args.seeds):
            data = make_data(args, d, fn, seed=s)          # built once, reused by all models
            for m in args.models:
                accs[m].append(run_one(m, data, args, seed=s)[0])
        row = f"  {d:<3d} "
        for m in args.models:
            row += f"{np.mean(accs[m])*100:10.1f}+-{np.std(accs[m])*100:4.1f}"
        print(row, flush=True)

    print("\n=== TEMPORAL RANDOM BALANCED 4-bit BOOLEAN (test accuracy %, mean +/- sd) ===")
    d = 4
    accs = {m: [] for m in args.models}
    entropies = {m: [] for m in args.models}
    for s in range(args.seeds):
        rng = np.random.default_rng(1000 + s)
        for r in range(args.n_random_rules):
            fn = random_balanced_boolean(d, rng)
            data = make_data(args, d, fn, seed=2000 + s * 100 + r)   # once per (seed, rule)
            for m in args.models:
                acc, _, ent = run_one(m, data, args, seed=2000 + s * 100 + r)
                accs[m].append(acc)
                if ent is not None:
                    entropies[m].append(ent)
    row = "  4   "
    for m in args.models:
        extra = f" H={np.mean(entropies[m]):.2f}" if entropies[m] else ""
        row += f"{np.mean(accs[m])*100:10.1f}+-{np.std(accs[m])*100:4.1f}{extra:>7s}"
    print(row)
    print("\nH is mean normalized routing entropy for routed temporal dendritic models.\n")


if __name__ == "__main__":
    main()
