"""Long-range coincidence benchmark: selective-SSM models vs conv baselines.

Spatiotemporal Boolean tasks where each bit is a timed spike burst. In
LONG-RANGE mode the evidence appears early and the readout is at the end of the
sequence, leaving a gap longer than any fixed conv kernel -- so only models with
genuine (unbounded, selective) memory can solve it.

Headline finding: a stacked coincidence-of-selective-memory block with a
regenerative plateau (`ssm_coinc`) solves long-range parity up to d=4, which the
conv-based temporal models provably cannot. Ablations show the plateau is
essential and the multiplicative coincidence gives the robust optimization path;
the dendritic tree topology (`dendritic_ssm`) does not help.

Models (all sized to the same per-block parameter budget):
  swiglu             pointwise reference (no temporal mixing)
  temporal_conv      causal depthwise conv + SwiGLU (memory, no coincidence)
  gated_conv         multiplicative conv coincidence (bounded memory)
  mamba              flat selective SSM (selective memory, no coincidence)
  ssm_coinc          two selective SSMs multiplied + plateau   (winner)
  ssm_sum            ablation: two selective SSMs ADDED (no coincidence)
  ssm_coinc_noplat   ablation: multiplicative coincidence, no plateau
  dendritic_ssm      negative control: branch tree + low-rank multiplicative soma

Examples:
    uv run --no-sync python -u experiments/run_ssm.py --preset smoke
    uv run --no-sync python -u experiments/run_ssm.py --preset longrange --device cuda
    # RTX 3080 version: bigger batch/samples/seeds, deeper, pushes d=5 by default:
    uv run --no-sync python -u experiments/run_ssm.py --preset gpu3080 \
        --device cuda --out results_3080.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from src.blocks import SwiGLU, CausalConvSwiGLU, GatedConvFFN, TemporalBlockClassifier
from src.counting import count_params, size_to_budget
from src.results_store import ResultStore
from src.ssm import MambaBlock, CoincidenceSSM, DendriticSSMBlock
from src.tasks import (parity_fn, random_balanced_boolean, subset_parity_fn,
                       make_temporal_boolean_dataset)
from src.train import pick_device, set_seed, train_classifier

# Fixed memory kernel for the conv reference baselines (the strong setting from
# the prior temporal sweep).
CONV_MEM = 9


@dataclass(frozen=True)
class BlockCfg:
    """Architecture knobs shared by every block in a sweep."""
    d_model: int
    d_state: int
    conv_k: int
    n_branches: int
    # Scan chunk for the SSM cores: bigger = fewer Python-loop iterations / kernel
    # launches (faster on GPU) at the cost of more peak memory. Math is identical.
    chunk: int = 8


@dataclass(frozen=True)
class ModelSpec:
    """How to build one model at a given search width, and what the width means."""
    build: Callable[[int, BlockCfg], nn.Module]
    width_tag: str


def _ssm_width(w: int) -> int:
    return max(2, w)


# The model universe is data, not a dispatch chain. Each builder maps a search
# width + shared BlockCfg to a block; `size_to_budget` picks the width.
REGISTRY: dict[str, ModelSpec] = {
    "swiglu": ModelSpec(lambda w, c: SwiGLU(c.d_model, w), "hidden"),
    "temporal_conv": ModelSpec(
        lambda w, c: CausalConvSwiGLU(c.d_model, w, memory_kernel=CONV_MEM), "hidden"),
    "gated_conv": ModelSpec(
        lambda w, c: GatedConvFFN(c.d_model, w, memory_kernel=CONV_MEM), "hidden"),
    "mamba": ModelSpec(
        lambda w, c: MambaBlock(c.d_model, d_inner=_ssm_width(w), d_state=c.d_state,
                                conv_k=c.conv_k, chunk=c.chunk), "d_inner"),
    "ssm_coinc": ModelSpec(
        lambda w, c: CoincidenceSSM(c.d_model, d_inner=_ssm_width(w), d_state=c.d_state,
                                    conv_k=c.conv_k, chunk=c.chunk), "d_inner"),
    "ssm_sum": ModelSpec(
        lambda w, c: CoincidenceSSM(c.d_model, d_inner=_ssm_width(w), d_state=c.d_state,
                                    conv_k=c.conv_k, combine="sum", chunk=c.chunk), "d_inner"),
    "ssm_coinc_noplat": ModelSpec(
        lambda w, c: CoincidenceSSM(c.d_model, d_inner=_ssm_width(w), d_state=c.d_state,
                                    conv_k=c.conv_k, use_plateau=False, chunk=c.chunk), "d_inner"),
    "dendritic_ssm": ModelSpec(
        lambda w, c: DendriticSSMBlock(c.d_model, d_inner=_ssm_width(w),
                                       n_branches=c.n_branches, d_state=c.d_state,
                                       conv_k=c.conv_k, chunk=c.chunk), "d_inner"),
    # Ablations that dissect WHY the dendritic tree works (it beat ssm_coinc at
    # d=4). Each drops one ingredient; if the win is "coincidence + plateau" then
    # _add and _noplat should fall to chance like ssm_sum / ssm_coinc_noplat, and
    # _notree should test whether the multi-branch topology itself matters.
    "dendritic_add": ModelSpec(            # drop the multiplicative soma binding
        lambda w, c: DendriticSSMBlock(c.d_model, d_inner=_ssm_width(w),
                                       n_branches=c.n_branches, d_state=c.d_state,
                                       conv_k=c.conv_k, chunk=c.chunk,
                                       soma_mode="add"), "d_inner"),
    "dendritic_noplat": ModelSpec(         # drop the regenerative plateau
        lambda w, c: DendriticSSMBlock(c.d_model, d_inner=_ssm_width(w),
                                       n_branches=c.n_branches, d_state=c.d_state,
                                       conv_k=c.conv_k, chunk=c.chunk,
                                       use_plateau=False), "d_inner"),
    "dendritic_notree": ModelSpec(         # collapse the tree to a single branch
        lambda w, c: DendriticSSMBlock(c.d_model, d_inner=_ssm_width(w),
                                       n_branches=c.n_branches, d_state=c.d_state,
                                       conv_k=c.conv_k, chunk=c.chunk,
                                       use_tree=False), "d_inner"),
    # Fixed branch-count variants for the tree dose-response (does reliability
    # rise with branch count?). n=1 is dendritic_notree, n=4 is dendritic_ssm at
    # the default branch budget; these fill in 2/8/16 at the same param budget.
    "dendritic_b2": ModelSpec(
        lambda w, c: DendriticSSMBlock(c.d_model, d_inner=_ssm_width(w),
                                       n_branches=2, d_state=c.d_state,
                                       conv_k=c.conv_k, chunk=c.chunk), "d_inner"),
    "dendritic_b8": ModelSpec(
        lambda w, c: DendriticSSMBlock(c.d_model, d_inner=_ssm_width(w),
                                       n_branches=8, d_state=c.d_state,
                                       conv_k=c.conv_k, chunk=c.chunk), "d_inner"),
    "dendritic_b16": ModelSpec(
        lambda w, c: DendriticSSMBlock(c.d_model, d_inner=_ssm_width(w),
                                       n_branches=16, d_state=c.d_state,
                                       conv_k=c.conv_k, chunk=c.chunk), "d_inner"),
}
MODELS = list(REGISTRY)


def sized_factory(name: str, target_params: int, bcfg: BlockCfg):
    """Return (make_block, width) for `name`, sized to the per-block budget."""
    spec = REGISTRY[name]
    width = size_to_budget(lambda w: spec.build(w, bcfg), target_params)
    return (lambda: spec.build(width, bcfg)), width


def make_data(args, d, fn, seed):
    active_frac = None
    if args.active_lo is not None and args.active_hi is not None:
        active_frac = (args.active_lo, args.active_hi)
    return make_temporal_boolean_dataset(
        d, fn, n_per_pattern=args.samples, n_time=args.n_time,
        axons_per_bit=args.axons_per_bit, jitter_std=args.jitter,
        spike_width=args.spike_width, background_rate=args.background_rate,
        seed=seed, active_frac=active_frac)


def train_cell(args, bcfg, model_name, data, seed):
    """Train one model on a pre-built dataset; return (test_acc, n_params)."""
    Xtr, ytr, Xte, yte = data
    set_seed(seed)
    make_block, _ = sized_factory(model_name, args.target_params, bcfg)
    model = TemporalBlockClassifier(
        d_in=Xtr.shape[-1], d_model=args.d_model, make_block=make_block,
        n_layers=args.n_layers, decision_window=args.decision_window)
    res = train_classifier(
        model, Xtr, ytr, Xte, yte, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, device=args.device)
    return res["best_test_acc"], count_params(make_block())


def run_group(args, bcfg, store, section, d, fn, seed):
    """Run every pending model for one (section, d, seed); build data lazily.

    Data is built once and only if some model in the group still needs running,
    so resumed sweeps skip both training and data generation for done cells.
    """
    pending = [m for m in args.models if not store.has(section, d, m, seed)]
    if not pending:
        return
    data = make_data(args, d, fn, seed)
    for m in pending:
        t0 = time.time()
        acc, params = train_cell(args, bcfg, m, data, seed)
        store.record(section, d, m, seed, acc, params, time.time() - t0)
        print(f"  [{section} d={d} seed={seed}] {m:16s} {acc*100:5.1f}%  "
              f"({time.time() - t0:5.0f}s)", flush=True)
        if args.device == "cuda":
            torch.cuda.empty_cache()
        if args.cooldown > 0:
            time.sleep(args.cooldown)


PRESETS = {
    "smoke": dict(
        d_list=[2, 4], target_params=20000, d_model=64, n_layers=1, epochs=30,
        lr=3e-3, batch_size=128, samples=12, n_time=24, axons_per_bit=4,
        decision_window=8, n_branches=4, d_state=8, conv_k=4, jitter=1.25,
        spike_width=0.5, background_rate=0.001, seeds=1, n_random_rules=2,
        active_lo=None, active_hi=None, chunk=8),
    "cpu": dict(
        d_list=[2, 4, 6], target_params=30000, d_model=96, n_layers=1, epochs=80,
        lr=3e-3, batch_size=256, samples=24, n_time=40, axons_per_bit=6,
        decision_window=10, n_branches=8, d_state=8, conv_k=4, jitter=1.5,
        spike_width=0.75, background_rate=0.002, seeds=2, n_random_rules=4,
        active_lo=None, active_hi=None, chunk=8),
    "gpu": dict(
        d_list=[4, 6, 8, 10], target_params=80000, d_model=128, n_layers=1,
        epochs=200, lr=2e-3, batch_size=256, samples=48, n_time=64,
        axons_per_bit=8, decision_window=12, n_branches=8, d_state=16, conv_k=4,
        jitter=1.5, spike_width=0.75, background_rate=0.002, seeds=3,
        n_random_rules=8, active_lo=None, active_hi=None, chunk=16),
    # The headline regime: spikes early, readout at the end (long gap). Override
    # --d-list / --n-layers / --seeds to push the frontier (e.g. d=5, 3 layers).
    "longrange": dict(
        d_list=[4], target_params=50000, d_model=96, n_layers=2, epochs=300,
        lr=2e-3, batch_size=256, samples=48, n_time=64, axons_per_bit=8,
        decision_window=12, n_branches=4, d_state=8, conv_k=4, jitter=1.5,
        spike_width=0.75, background_rate=0.002, seeds=3, n_random_rules=0,
        active_lo=0.0, active_hi=0.35, chunk=16),
    # RTX 3080 (10GB Ampere) long-range version. The SSM scan is a Python loop,
    # so it is launch-bound, not FLOP-bound: a big batch does NOT make the SSM
    # models proportionally faster, it just makes each cell take minutes. So we
    # keep the batch modest, raise `chunk` (fewer loop iterations = the real GPU
    # speedup), and trim epochs/seeds so the whole d=4+d=5 sweep finishes in a
    # few hours. TF32 + cuDNN autotune turn on automatically on CUDA. Scale up
    # with --epochs / --seeds / --samples once you've seen the per-cell timing.
    "gpu3080": dict(
        d_list=[4, 5], target_params=60000, d_model=128, n_layers=3, epochs=200,
        lr=2e-3, batch_size=256, samples=64, n_time=64, axons_per_bit=8,
        decision_window=12, n_branches=4, d_state=16, conv_k=4, jitter=1.5,
        spike_width=0.75, background_rate=0.002, seeds=2, n_random_rules=0,
        active_lo=0.0, active_hi=0.35, chunk=32),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=list(PRESETS), default="smoke")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default="results_ssm_runs.csv",
                    help="CSV for incremental, resumable per-cell results")
    ap.add_argument("--cooldown", type=float, default=0.0,
                    help="seconds to sleep between trainings (thermal headroom)")
    ap.add_argument("--threads", type=int, default=0,
                    help="cap torch CPU threads (0=default); lower reduces sustained "
                         "CPU power draw on thermally-limited machines")
    ap.add_argument("--solved-thresh", type=float, default=0.9,
                    help="accuracy above which a seed counts as having 'solved' the "
                         "task, for the fraction-solved summary")
    ap.add_argument("--subset-k", type=int, nargs="+", default=None,
                    help="if set, also run SUBSET PARITY for each k: parity over the "
                         "first k of d bits with d-k distractors (tests selective "
                         "coincidence). Pass several values to sweep coincidence order, "
                         "e.g. --subset-k 2 3 4 5")
    # Everything below defaults to None and is filled from the chosen preset.
    for name, typ in [("d-list", int), ("target-params", int), ("d-model", int),
                      ("n-layers", int), ("epochs", int), ("lr", float),
                      ("batch-size", int), ("samples", int), ("n-time", int),
                      ("axons-per-bit", int), ("decision-window", int),
                      ("n-branches", int), ("d-state", int), ("conv-k", int),
                      ("jitter", float), ("spike-width", float),
                      ("background-rate", float), ("seeds", int),
                      ("n-random-rules", int), ("active-lo", float),
                      ("active-hi", float), ("chunk", int)]:
        ap.add_argument(f"--{name}", type=typ, nargs="+" if name == "d-list" else None,
                        default=None)
    return ap.parse_args()


def configure(args):
    """Apply thread cap, fill unset args from the preset, resolve the device."""
    if args.threads > 0:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(max(1, args.threads // 2))
        except RuntimeError:
            pass  # interop threads can only be set once per process
        print(f"Capped torch to {args.threads} CPU threads (of {os.cpu_count()}).")

    for key, val in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    unknown = [m for m in args.models if m not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    args.device = pick_device(args.device)
    if args.device == "cuda":
        # Ampere (e.g. RTX 3080) freebies: TF32 matmuls and cuDNN autotuning for
        # the fixed conv shapes. Pure speed, no behaviour change worth caring about.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return BlockCfg(d_model=args.d_model, d_state=args.d_state,
                    conv_k=args.conv_k, n_branches=args.n_branches,
                    chunk=args.chunk)


def print_header(args, bcfg):
    span = "full"
    if args.active_lo is not None and args.active_hi is not None:
        span = (f"LONG-RANGE spikes in [{args.active_lo:.2f},{args.active_hi:.2f}]*T, "
                f"gap before readout (decision_window={args.decision_window})")
    print(f"\nDevice: {args.device} (preset={args.preset}). T={args.n_time}, "
          f"axons/bit={args.axons_per_bit}, jitter={args.jitter}, "
          f"branches={args.n_branches}, d_state={args.d_state}, scan_chunk={args.chunk}, "
          f"budget~{args.target_params} params/block.\nSpike timing: {span}.\n")
    print("Block sizes:")
    for m in args.models:
        make_block, width = sized_factory(m, args.target_params, bcfg)
        spec = REGISTRY[m]
        print(f"  {m:18s} {spec.width_tag}={width:<5d} "
              f"actual_params={count_params(make_block())}")


def print_table(title, args, store, section, rows):
    """rows: list of (d, seed_list) to summarize as mean +/- sd per model."""
    print(f"\n=== {title} (test accuracy %, mean +/- sd) ===")
    print("  d   " + "".join(f"{m:>18s}" for m in args.models))
    for d, seeds in rows:
        line = f"  {d:<3d} "
        for m in args.models:
            vals = store.values(section, d, m, seeds)
            line += (f"{np.mean(vals)*100:11.1f}+-{np.std(vals)*100:4.1f}"
                     if vals else f"{'--':>18s}")
        print(line, flush=True)

    # In this regime accuracy is bimodal (a seed either finds the coincidence
    # solution or sits at chance), so "how many seeds solved it" is the honest
    # summary. A model that solves k/n seeds is doing something the chance
    # models (0/n) provably cannot, even if its mean looks middling.
    print(f"  -- fraction of seeds solved (acc > {args.solved_thresh:.2f}) --")
    for d, seeds in rows:
        line = f"  {d:<3d} "
        for m in args.models:
            vals = store.values(section, d, m, seeds)
            if vals:
                k = sum(v > args.solved_thresh for v in vals)
                line += f"{k}/{len(vals):<16d}"
            else:
                line += f"{'--':>18s}"
        print(line, flush=True)


def main():
    args = parse_args()
    bcfg = configure(args)
    print_header(args, bcfg)

    store = ResultStore(args.out)
    if len(store):
        print(f"Resuming: {len(store)} cells already in {args.out} will be skipped.")

    parity_seeds = list(range(args.seeds))
    print("\n=== TEMPORAL PARITY (running; per-cell results saved) ===", flush=True)
    for d in args.d_list:
        fn = parity_fn(d)
        for s in parity_seeds:
            run_group(args, bcfg, store, "parity", d, fn, s)
    print_table("TEMPORAL PARITY", args, store, "parity",
                [(d, parity_seeds) for d in args.d_list])

    if args.n_random_rules > 0:
        # One random rule per seed; the seed encodes (base seed, rule index).
        rand = []  # (seed, fn)
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 + s)
            for r in range(args.n_random_rules):
                rand.append((2000 + s * 100 + r, random_balanced_boolean(4, rng)))
        print("\n=== TEMPORAL RANDOM BALANCED 4-bit (running) ===", flush=True)
        for seed, fn in rand:
            run_group(args, bcfg, store, "rand", 4, fn, seed)
        print_table("TEMPORAL RANDOM BALANCED 4-bit", args, store, "rand",
                    [(4, [seed for seed, _ in rand])])

    for k in (args.subset_k or []):
        section = f"subset_k{k}"
        sub_rows = []
        print(f"\n=== SUBSET PARITY k={k} (parity of {k} bits, rest distractors; "
              f"running) ===", flush=True)
        for d in args.d_list:
            if d <= k:
                continue  # need at least one distractor bit
            fn = subset_parity_fn(d, k)
            for s in parity_seeds:
                run_group(args, bcfg, store, section, d, fn, s)
            sub_rows.append((d, parity_seeds))
        if sub_rows:
            print_table(f"SUBSET PARITY (k={k})", args, store, section, sub_rows)
    print()


if __name__ == "__main__":
    main()
