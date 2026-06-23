"""Streaming parity (running XOR): an autoregressive STATE-TRACKING test.

At each timestep the model reads a bit and must output the parity of all bits so
far. This is the autoregressive analogue of the long-range coincidence tasks:
it asks whether the dendritic block's plateau x multiplicative coincidence buys
genuine state-tracking that a flat selective SSM (Mamba) or attention lack --
and, crucially, whether it LENGTH-GENERALIZES (train short, test long), which is
the honest discriminator. A model that only memorizes a count->parity readout
fits the training length but degrades as sequences grow; a model that maintains
a bounded parity state holds up.

All models share the same scaffold (pre-norm token-mixer + SwiGLU FFN, per-step
binary head); only the TOKEN MIXER is swapped, at a matched per-mixer parameter
budget (the SSM mixers are sized to the attention mixer's param count):

  attention      causal self-attention (with learned positions)
  mamba          flat selective SSM (linear recurrence)
  ssm_coinc      two selective-memory streams multiplied + plateau
  dendritic_ssm  tree of plateau-gated selective branches + multiplicative soma

Usage:
    uv run --no-sync python -u experiments/run_streaming_parity.py --preset cpu
    uv run --no-sync python -u experiments/run_streaming_parity.py --preset gpu3080 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from src.blocks import SwiGLU
from src.counting import count_params, size_to_budget
from src.ssm import MambaBlock, CoincidenceSSM, DendriticSSMBlock
from src.tasks import make_streaming_parity
from src.train import pick_device, set_seed


class CausalAttention(nn.Module):
    """Standard causal multi-head self-attention as a (B, T, C) -> (B, T, C) mixer."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = (B, T, self.h, C // self.h)
        q, k, v = (t.view(*shape).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


@dataclass(frozen=True)
class MixerSpec:
    """How to build a token mixer at a given width, and whether it needs positions.

    SSM mixers are inherently positional (the recurrence carries order), so they
    skip the learned position embedding; attention needs it.
    """
    build: Callable[[int, "MixerCfg"], nn.Module]
    needs_pos: bool


@dataclass(frozen=True)
class MixerCfg:
    d_model: int
    n_heads: int
    d_state: int
    conv_k: int
    n_branches: int
    chunk: int


def _w(w: int) -> int:
    return max(2, w)


MIXERS: dict[str, MixerSpec] = {
    "attention": MixerSpec(lambda w, c: CausalAttention(c.d_model, c.n_heads), True),
    "mamba": MixerSpec(
        lambda w, c: MambaBlock(c.d_model, d_inner=_w(w), d_state=c.d_state,
                                conv_k=c.conv_k, chunk=c.chunk), False),
    "ssm_coinc": MixerSpec(
        lambda w, c: CoincidenceSSM(c.d_model, d_inner=_w(w), d_state=c.d_state,
                                    conv_k=c.conv_k, chunk=c.chunk), False),
    "dendritic_ssm": MixerSpec(
        lambda w, c: DendriticSSMBlock(c.d_model, d_inner=_w(w), n_branches=c.n_branches,
                                       d_state=c.d_state, conv_k=c.conv_k,
                                       chunk=c.chunk), False),
}
MODELS = list(MIXERS)


def sized_mixer(name: str, target_params: int, cfg: MixerCfg):
    """Return (make_mixer, needs_pos). Attention is fixed by d_model; the SSM
    mixers are binary-searched to the same per-mixer param budget."""
    spec = MIXERS[name]
    if name == "attention":
        return (lambda: spec.build(0, cfg)), spec.needs_pos
    width = size_to_budget(lambda w: spec.build(w, cfg), target_params)
    return (lambda: spec.build(width, cfg)), spec.needs_pos


class Block(nn.Module):
    """Pre-norm token mixer + pre-norm SwiGLU FFN, residual around each."""

    def __init__(self, d_model: int, make_mixer: Callable[[], nn.Module], ffn_hidden: int):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.mix = make_mixer()
        self.n2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x


class ParityTagger(nn.Module):
    """Bit-stream -> per-position parity logit."""

    def __init__(self, d_model: int, n_layers: int, make_mixer: Callable[[], nn.Module],
                 ffn_hidden: int, max_len: int, use_pos: bool):
        super().__init__()
        self.emb = nn.Embedding(2, d_model)
        self.pos = nn.Embedding(max_len, d_model) if use_pos else None
        self.blocks = nn.ModuleList(
            Block(d_model, make_mixer, ffn_hidden) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        x = self.emb(bits)
        if self.pos is not None:
            T = bits.shape[1]
            if T > self.pos.num_embeddings:
                raise ValueError(f"seq_len {T} exceeds max_len {self.pos.num_embeddings}")
            x = x + self.pos(torch.arange(T, device=bits.device))[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x)).squeeze(-1)        # (B, T) logits


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
    max_len = max(args.eval_lens + [args.train_len])
    model = ParityTagger(args.d_model, args.n_layers, make_mixer,
                         args.ffn_mult * args.d_model, max_len, use_pos).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    model.train()
    for step in range(args.steps):
        bits, par = batch(rng, args.batch_size, args.train_len, args.device)
        loss = loss_fn(model(bits), par)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    accs = {L: evaluate(model, L, args.device) for L in args.eval_lens}
    return accs, count_params(model.blocks[0].mix)


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

    seeds = list(range(args.seeds))
    # acc[model][L] = list of (per_pos, final) over seeds
    results: dict[str, dict[int, list]] = {m: {L: [] for L in args.eval_lens}
                                           for m in args.models}
    print("\n=== TRAINING (per-position acc / final-position acc, by eval length) ===")
    for m in args.models:
        for s in seeds:
            accs, mp = train_one(m, cfg, args, target_params, s)
            for L, (pp, fin) in accs.items():
                results[m][L].append((pp, fin))
            shown = "  ".join(f"L{L}:{accs[L][0]*100:4.0f}/{accs[L][1]*100:4.0f}"
                              for L in args.eval_lens)
            print(f"  [{m:14s} seed={s}] {shown}", flush=True)

    print("\n=== STREAMING PARITY (per-position acc %, mean over seeds) ===")
    print("  model           " + "".join(f"{'L='+str(L):>10s}" for L in args.eval_lens))
    for m in args.models:
        line = f"  {m:14s}  "
        for L in args.eval_lens:
            pp = np.mean([v[0] for v in results[m][L]]) * 100
            line += f"{pp:>10.1f}"
        print(line)
    print("\n=== final-position acc % (the hard bit: parity of the whole stream) ===")
    print("  model           " + "".join(f"{'L='+str(L):>10s}" for L in args.eval_lens))
    for m in args.models:
        line = f"  {m:14s}  "
        for L in args.eval_lens:
            fin = np.mean([v[1] for v in results[m][L]]) * 100
            line += f"{fin:>10.1f}"
        print(line)
    print()


if __name__ == "__main__":
    main()
