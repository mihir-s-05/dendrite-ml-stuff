"""Streaming parity (running XOR): an autoregressive STATE-TRACKING test.

At each timestep the model reads a bit and must output the parity of all bits so
far. This is the autoregressive analogue of the long-range coincidence tasks:
it asks whether the dendritic block's plateau x multiplicative coincidence buys
genuine state-tracking that a flat selective SSM (Mamba) or attention lack --
and, crucially, whether it LENGTH-GENERALIZES (train short, test long), which is
the honest discriminator. A model that only memorizes a count->parity readout
fits the training length but degrades as sequences grow; a model that maintains
a bounded parity state holds up.

The model scaffold (mixer zoo, sizing, per-position tagger) is shared with the
mod-k experiment and lives in ``src.seq_tagger``; this file owns the parity
task, the binary loss, and the length-generalization reporting.

Usage:
    uv run --no-sync python -u experiments/run_streaming_parity.py --preset cpu
    uv run --no-sync python -u experiments/run_streaming_parity.py --preset gpu3080 --device cuda
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
from src.tasks import make_streaming_parity
from src.train import pick_device, set_seed


def batch(rng: np.random.Generator, bs: int, seq_len: int, device: str):
    bits, par = make_streaming_parity(bs, seq_len, seed=int(rng.integers(1 << 31)))
    return (torch.from_numpy(bits).to(device), torch.from_numpy(par).to(device))


@torch.no_grad()
def evaluate(model, seq_len: int, device: str, n: int = 2048, seed: int = 999,
             token_budget: int = 16384):
    """Per-position and final-position accuracy on a fixed eval set.

    Batched with a constant token budget (rows*seq_len) so peak memory stays flat
    across eval lengths -- the SSM scan materializes a (rows, L, d_inner, d_state)
    tensor, which OOMs at long L if the whole eval set is run in one pass.
    """
    model.eval()
    bits_all, par_all = make_streaming_parity(n, seq_len, seed=seed)
    eval_bs = max(8, min(n, token_budget // seq_len))
    pp_correct = pp_total = fin_correct = fin_total = 0
    for i in range(0, n, eval_bs):
        bits = torch.from_numpy(bits_all[i:i + eval_bs]).to(device)
        par = torch.from_numpy(par_all[i:i + eval_bs]).to(device)
        pred = (model(bits) > 0).float()
        pp_correct += (pred == par).float().sum().item()
        pp_total += par.numel()
        fin_correct += (pred[:, -1] == par[:, -1]).float().sum().item()
        fin_total += par.shape[0]
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
    args.device = pick_device(args.device)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return MixerCfg(d_model=args.d_model, n_heads=args.n_heads, d_state=args.d_state,
                    conv_k=args.conv_k, n_branches=args.n_branches, chunk=args.chunk)


def train_one(name, cfg, args, target_params, seed):
    set_seed(seed)
    make_mixer, use_pos = sized_mixer(name, target_params, cfg)
    train_lens = train_lengths(args)
    max_len = max(args.eval_lens + train_lens)
    model = SeqTagger(args.d_model, args.n_layers, make_mixer,
                      args.ffn_mult * args.d_model, max_len, use_pos, n_out=1).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    model.train()
    for step in range(args.steps):
        L_t = sample_train_len(train_lens, rng)
        bits, par = batch(rng, args.batch_size, L_t, args.device)
        loss = loss_fn(model(bits), par)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    accs = {L: evaluate(model, L, args.device) for L in args.eval_lens}
    return accs, count_params(model.blocks[0].mix), model


def main():
    args = parse_args()
    cfg = configure(args)

    # Budget = the attention mixer's param count; SSM mixers are sized to match.
    target_params = count_params(CausalAttention(args.d_model, args.n_heads))
    print(f"\nDevice: {args.device} (preset={args.preset}). "
          f"d_model={args.d_model} x{args.n_layers}L, train_len={args.train_len}, "
          f"eval_lens={args.eval_lens}, per-mixer budget~{target_params}, "
          f"seeds={args.seeds}.\n")
    print("Mixer sizes:")
    for m in args.models:
        make_mixer, _ = sized_mixer(m, target_params, cfg)
        print(f"  {m:14s} mixer_params={count_params(make_mixer())}")

    run_sweep(args, cfg, target_params, train_one, task_tag="parity",
              stream_title="STREAMING PARITY", chance=50.0,
              hard_desc="parity of the whole stream")


if __name__ == "__main__":
    main()
