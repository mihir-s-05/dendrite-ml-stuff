"""Tiny char-level transformer where the FFN sub-layer is swappable.

Compares MLP / SwiGLU / Dendritic FFN at a matched per-block parameter budget
on a small language-modeling task, reporting validation loss (bits/char). This
is the "does it help inside a real transformer" test; the headline scientific
signal is expected to be on the Boolean suite (run_boolean.py), with LM as a
sanity check that the block is competitive at matched compute.

Usage:
    python experiments/run_lm.py --steps 1500
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from src.blocks import MLP, SwiGLU, DendriticFFN
from src.counting import count_params, size_to_budget
from src.train import set_seed, pick_device

CORPUS = (
    "dendrites are not just wires; they are the substrate for nonlinear computation. "
    "a single cortical pyramidal neuron binds features across its tree. "
    "the apical tuft and the basal dendrites compute and-not operations, "
    "and a calcium spike combines them at the soma to implement exclusive or. "
    "nmda receptors act as coincidence detectors, a multiplicative gate. "
    "point neurons sum and threshold; dendritic neurons gate and bind. "
) * 40


def make_block_factory(name, d_model, target_params):
    if name == "mlp":
        h = size_to_budget(lambda h: MLP(d_model, h), target_params)
        return lambda: MLP(d_model, h), h
    if name == "swiglu":
        h = size_to_budget(lambda h: SwiGLU(d_model, h), target_params)
        return lambda: SwiGLU(d_model, h), h
    if name == "dendritic":
        K = size_to_budget(lambda K: DendriticFFN(d_model, n_branches=max(1, K), branch_dim=8),
                           target_params, lo=1, hi=2048)
        return lambda: DendriticFFN(d_model, n_branches=max(1, K), branch_dim=8), K
    raise ValueError(name)


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, make_ffn):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.n2 = nn.LayerNorm(d_model)
        self.ffn = make_ffn()

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, d_model, n_layers, n_heads, block_size, make_ffn):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, make_ffn) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix]).to(device)
    return x, y


# Shape/size presets. CPU = tiny + quick; GPU = larger run sized for an
# 8GB card (e.g. RTX 4070 mobile). Run-control flags left as None on the CLI
# fall back to the preset; pass them explicitly to override.
PRESETS = {
    "cpu": dict(d_model=64, n_layers=2, n_heads=4, block_size=64,
                batch_size=32, target_params=20000, steps=1500, lr=3e-3),
    "gpu": dict(d_model=256, n_layers=4, n_heads=8, block_size=128,
                batch_size=64, target_params=200000, steps=4000, lr=3e-3),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="cpu")
    ap.add_argument("--target-params", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    for key, val in cfg.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    args.device = pick_device(args.device)
    print(f"Device: {args.device}  (preset={args.preset})")

    chars = sorted(set(CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in CORPUS], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    print(f"\nvocab={len(chars)}  budget~{args.target_params} params/FFN  "
          f"d_model={args.d_model}  layers={args.n_layers}\n")

    for name in ["mlp", "swiglu", "dendritic"]:
        set_seed(0)
        make_ffn, h = make_block_factory(name, args.d_model, args.target_params)
        ffn_params = count_params(make_ffn())
        model = GPT(len(chars), args.d_model, args.n_layers, args.n_heads,
                    args.block_size, make_ffn).to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for step in range(args.steps):
            x, y = get_batch(train_data, args.block_size, args.batch_size, args.device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vlosses = []
            for _ in range(20):
                x, y = get_batch(val_data, args.block_size, args.batch_size, args.device)
                logits = model(x)
                vlosses.append(F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)).item())
        vloss = sum(vlosses) / len(vlosses)
        print(f"  {name:10s}  ffn_params={ffn_params:<7d}  "
              f"total_params={count_params(model):<8d}  "
              f"val_loss={vloss:.3f}  val_bpc={vloss/math.log(2):.3f}")
    print()


if __name__ == "__main__":
    main()
