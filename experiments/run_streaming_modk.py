"""Streaming mod-k counter: the MULTI-BIT generalization of streaming parity.

At each timestep the model reads a {0,1} increment and must output the running
count mod k. Parity is the k=2 special case (1 bit of state); k>2 needs a
k-state cyclic automaton, i.e. MORE than one bit of carried state, which a single
sign-flip recurrence cannot represent. This is the honest test of whether the
"regenerative nonlinearity in the loop" (the dendritic_rec block) is a general
length-invariant state-tracking mechanism or merely a parity-specific trick: a
true state machine length-generalizes (train short, test long); a count->readout
memorizer fits the train length and decays to chance (1/k) as sequences grow.

Shares the model scaffold with the parity experiment (``src.seq_tagger``); this
file owns the mod-k task, the multi-class loss, and the reporting.

Usage:
    uv run --no-sync python -u experiments/run_streaming_modk.py --preset cpu
    uv run --no-sync python -u experiments/run_streaming_modk.py --preset gpu3080 --device cuda --mod 3
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from src.counting import count_params
from src.seq_tagger import MIXERS, MODELS, CausalAttention, MixerCfg, SeqTagger, sized_mixer
from src.streaming_sweep import (add_sweep_args, run_sweep, sample_train_len,
                                 train_lengths)
from src.tasks import make_streaming_modk
from src.train import pick_device, set_seed


def batch(rng: np.random.Generator, bs: int, seq_len: int, k: int, device: str):
    bits, y = make_streaming_modk(bs, seq_len, k=k, seed=int(rng.integers(1 << 31)))
    return (torch.from_numpy(bits).to(device), torch.from_numpy(y).to(device))


@torch.no_grad()
def evaluate(model, seq_len: int, k: int, device: str, n: int = 2048, seed: int = 999,
             token_budget: int = 16384):
    """Per-position and final-position accuracy on a fixed eval set.

    Batched with a constant token budget so peak memory stays flat across eval
    lengths (the SSM scan materializes a (rows, L, d_inner, d_state) tensor).
    """
    model.eval()
    bits_all, y_all = make_streaming_modk(n, seq_len, k=k, seed=seed)
    eval_bs = max(8, min(n, token_budget // seq_len))
    pp_correct = pp_total = fin_correct = fin_total = 0
    for i in range(0, n, eval_bs):
        bits = torch.from_numpy(bits_all[i:i + eval_bs]).to(device)
        y = torch.from_numpy(y_all[i:i + eval_bs]).to(device)
        pred = model(bits).argmax(dim=-1)                 # (B, T)
        pp_correct += (pred == y).float().sum().item()
        pp_total += y.numel()
        fin_correct += (pred[:, -1] == y[:, -1]).float().sum().item()
        fin_total += y.shape[0]
    return pp_correct / pp_total, fin_correct / fin_total


PRESETS = {
    "cpu": dict(d_model=64, n_layers=2, n_heads=4, train_len=24,
                eval_lens=[24, 48, 96], batch_size=64, steps=2000, lr=3e-3,
                ffn_mult=2, d_state=16, conv_k=4, n_branches=4, chunk=16, seeds=1),
    "gpu3080": dict(d_model=128, n_layers=3, n_heads=8, train_len=32,
                    eval_lens=[32, 64, 128, 256], batch_size=128, steps=8000,
                    lr=2e-3, ffn_mult=2, d_state=16, conv_k=4, n_branches=4,
                    chunk=32, seeds=3),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=list(PRESETS), default="cpu")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--mod", type=int, default=3, help="counter modulus k (k=2 is parity)")
    ap.add_argument("--rot-bins", type=int, default=None,
                    help="angle-grid size for dendritic_qrot (default: 4*mod, a multiple "
                         "of k so the exact 2*pi/k angle is representable)")
    ap.add_argument("--snap-warmup-frac", type=float, default=0.5,
                    help="dendritic_qrot: fraction of training over which the angle snap "
                         "is annealed soft->hard (0 = hard from step 0)")
    add_sweep_args(ap)
    for name, typ in [("d-model", int), ("n-layers", int), ("n-heads", int),
                      ("train-len", int), ("batch-size", int), ("steps", int),
                      ("lr", float), ("ffn-mult", int), ("d-state", int),
                      ("conv-k", int), ("n-branches", int), ("chunk", int),
                      ("seeds", int)]:
        ap.add_argument(f"--{name}", type=typ, default=None)
    ap.add_argument("--eval-lens", type=int, nargs="+", default=None)
    return ap.parse_args()


def configure(args):
    if args.threads > 0:
        torch.set_num_threads(args.threads)
        print(f"Capped torch to {args.threads} CPU threads (of {os.cpu_count()}).")
    for key, val in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    unknown = [m for m in args.models if m not in MIXERS]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    if args.mod < 2:
        raise ValueError(f"--mod must be >= 2, got {args.mod}")
    if args.rot_bins is None:
        args.rot_bins = 4 * args.mod
    args.device = pick_device(args.device)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return MixerCfg(d_model=args.d_model, n_heads=args.n_heads, d_state=args.d_state,
                    conv_k=args.conv_k, n_branches=args.n_branches, chunk=args.chunk,
                    rot_bins=args.rot_bins)


def train_one(name, cfg, args, target_params, seed):
    set_seed(seed)
    make_mixer, use_pos = sized_mixer(name, target_params, cfg)
    train_lens = train_lengths(args)
    max_len = max(args.eval_lens + train_lens)
    model = SeqTagger(args.d_model, args.n_layers, make_mixer,
                      args.ffn_mult * args.d_model, max_len, use_pos,
                      n_out=args.mod).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)

    def set_snap(val):  # ramp the qrot angle-snap hardness (no-op for other models)
        for mod in model.modules():
            if hasattr(mod, "snap_alpha"):
                mod.snap_alpha.fill_(val)

    warmup = max(1, int(args.snap_warmup_frac * args.steps))
    model.train()
    for step in range(args.steps):
        set_snap(min(1.0, step / warmup))
        L_t = sample_train_len(train_lens, rng)
        bits, y = batch(rng, args.batch_size, L_t, args.mod, args.device)
        logits = model(bits)                              # (B, T, k)
        loss = loss_fn(logits.reshape(-1, args.mod), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    set_snap(1.0)                                         # full exact snap for eval
    accs = {L: evaluate(model, L, args.mod, args.device) for L in args.eval_lens}
    return accs, count_params(model.blocks[0].mix), model


def main():
    args = parse_args()
    cfg = configure(args)

    target_params = count_params(CausalAttention(args.d_model, args.n_heads))
    print(f"\nDevice: {args.device} (preset={args.preset}). mod k={args.mod} "
          f"(chance={100.0/args.mod:.1f}%). d_model={args.d_model} x{args.n_layers}L, "
          f"train_len={args.train_len}, eval_lens={args.eval_lens}, "
          f"per-mixer budget~{target_params}, rot_bins={args.rot_bins}, "
          f"seeds={args.seeds}.\n")
    print("Mixer sizes:")
    for m in args.models:
        make_mixer, _ = sized_mixer(m, target_params, cfg)
        print(f"  {m:14s} mixer_params={count_params(make_mixer())}")

    run_sweep(args, cfg, target_params, train_one, task_tag=f"mod {args.mod}",
              stream_title=f"STREAMING MOD-{args.mod}", chance=100.0 / args.mod,
              hard_desc=f"count mod {args.mod} of the whole stream")


if __name__ == "__main__":
    main()
