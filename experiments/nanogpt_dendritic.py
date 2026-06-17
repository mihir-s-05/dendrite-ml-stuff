"""A nanoGPT-style char transformer whose FFN is the context-routed
DendriticGatedFFN, tested where the architecture should actually help:
CONTINUAL multi-domain language modeling (train on domains one after another,
no replay) and measure catastrophic forgetting vs a vanilla SwiGLU FFN.

Domains have distinct structure/char-statistics (english / code / math / dna)
over a shared vocab, so sequential training induces interference.

Metrics (bits/char on each domain's val set; lower is better):
  - final mean bpc : average over domains after the whole sequence.
  - forgetting     : mean over earlier domains of (bpc at end - bpc right
                     after that domain was trained); higher = forgot more.

Usage:
    uv run --no-sync python -u experiments/nanogpt_dendritic.py --device cuda
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from src.blocks import DendriticGatedFFN, SwiGLUFFN
from src.counting import count_params
from src.train import set_seed, pick_device


# ----------------------------- multi-domain data -----------------------------
def gen_english(n, rng):
    base = ("dendrites are not just wires they are the substrate for nonlinear "
            "computation a single cortical neuron binds features across its tree "
            "the apical tuft and basal dendrites compute and gate signals while "
            "nmda receptors act as coincidence detectors a multiplicative gate ")
    return (base * (n // len(base) + 1))[:n]


def gen_code(n, rng):
    lines = [
        "def add(a, b): return a + b",
        "for i in range(10): print(i * 2)",
        "if x > 0 and y < 5: z = x * y - 1",
        "class Node: def __init__(self): self.next = None",
        "while count < limit: count += step",
    ]
    out = []
    while sum(len(s) for s in out) < n:
        out.append(rng.choice(lines) + "\n")
    return "".join(out)[:n]


def gen_math(n, rng):
    out = []
    while sum(len(s) for s in out) < n:
        a, b = rng.integers(0, 100), rng.integers(0, 100)
        op = rng.choice(["+", "-", "*"])
        val = a + b if op == "+" else a - b if op == "-" else a * b
        out.append(f"{a}{op}{b}={val}; ")
    return "".join(out)[:n]


def gen_dna(n, rng):
    motifs = ["ACGT", "TATA", "GGGCCC", "AATT", "CGCG"]
    out = []
    while sum(len(s) for s in out) < n:
        out.append("".join(rng.choice(list("ACGT"), size=rng.integers(3, 8))))
        if rng.random() < 0.3:
            out.append(rng.choice(motifs))
        out.append(" ")
    return "".join(out)[:n]


DOMAINS = {"english": gen_english, "code": gen_code, "math": gen_math, "dna": gen_dna}


def build_corpora(n_chars, seed):
    rng = np.random.default_rng(seed)
    texts = {name: fn(n_chars, rng) for name, fn in DOMAINS.items()}
    vocab = sorted(set("".join(texts.values())))
    stoi = {c: i for i, c in enumerate(vocab)}
    data = {}
    for name, t in texts.items():
        ids = torch.tensor([stoi[c] for c in t], dtype=torch.long)
        n = int(0.9 * len(ids))
        data[name] = (ids[:n], ids[n:])
    return data, len(vocab)


# ------------------------------- model ---------------------------------------
class Attention(nn.Module):
    def __init__(self, d_model, n_heads, **_):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def _attn(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = (B, T, self.h, C // self.h)
        q, k, v = (t.view(*shape).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return y.transpose(1, 2).contiguous().view(B, T, C)

    def forward(self, x, ctx=None):
        return self.proj(self._attn(x))


class DendriticGatedAttention(Attention):
    """Attention whose per-channel output is context-routed (dendritic gate)
    before the output projection, so different domains use different attention
    subspaces and downstream weights forget less."""

    def __init__(self, d_model, n_heads, n_ctx, n_segments=8, dend_init=1.5, **_):
        super().__init__(d_model, n_heads)
        self.dend = nn.Parameter(torch.randn(d_model, n_segments, n_ctx) * dend_init)

    def forward(self, x, ctx):
        y = self._attn(x)
        a = torch.einsum("bc,fsc->bfs", ctx, self.dend)
        winner = torch.gather(a, -1, a.abs().argmax(-1, keepdim=True)).squeeze(-1)
        gate = torch.sigmoid(winner).unsqueeze(1)         # (B, 1, C) over T
        return self.proj(y * gate)


class Block(nn.Module):
    def __init__(self, attn, ffn):
        super().__init__()
        d_model = attn.proj.out_features
        self.n1 = nn.LayerNorm(d_model)
        self.attn = attn
        self.n2 = nn.LayerNorm(d_model)
        self.ffn = ffn

    def forward(self, x, ctx):
        x = x + self.attn(self.n1(x), ctx)
        x = x + self.ffn(self.n2(x), ctx)
        return x


class NanoGPT(nn.Module):
    def __init__(self, vocab, d_model, n_layers, n_heads, block_size,
                 make_ffn, make_attn):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(make_attn(), make_ffn())
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx, ctx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for b in self.blocks:
            x = b(x, ctx)
        return self.head(self.norm(x))


# ------------------------------ train / eval ---------------------------------
def get_batch(data, block_size, bs, device):
    ix = torch.randint(len(data) - block_size - 1, (bs,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix]).to(device)
    return x, y


def onehot(d, n, rows, device):
    v = torch.zeros(rows, n, device=device)
    v[:, d] = 1.0
    return v


@torch.no_grad()
def eval_bpc(model, val, d, n_dom, block_size, bs, device, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(val, block_size, bs, device)
        ctx = onehot(d, n_dom, len(x), device)
        loss = F.cross_entropy(model(x, ctx).view(-1, model.head.out_features),
                               y.view(-1))
        losses.append(loss.item())
    return float(np.mean(losses)) / math.log(2)


def train_domain(model, train, d, n_dom, steps, block_size, bs, lr, device):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        x, y = get_batch(train, block_size, bs, device)
        ctx = onehot(d, n_dom, len(x), device)
        loss = F.cross_entropy(model(x, ctx).view(-1, model.head.out_features),
                               y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()


def run(name, make_ffn, make_attn, data, vocab, args, device):
    set_seed(args.seed)
    model = NanoGPT(vocab, args.d_model, args.n_layers, args.n_heads,
                    args.block_size, make_ffn, make_attn).to(device)
    names = list(data)
    bpc_after = {}
    for d, dom in enumerate(names):
        train_domain(model, data[dom][0], d, len(names), args.steps,
                     args.block_size, args.batch, args.lr, device)
        bpc_after[dom] = eval_bpc(model, data[dom][1], d, len(names),
                                  args.block_size, args.batch, device)
    bpc_final = {dom: eval_bpc(model, data[dom][1], d, len(names),
                               args.block_size, args.batch, device)
                 for d, dom in enumerate(names)}
    forgetting = float(np.mean([bpc_final[dom] - bpc_after[dom]
                                for dom in names[:-1]]))
    return count_params(model), bpc_final, float(np.mean(list(bpc_final.values()))), forgetting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n-chars", type=int, default=20000)
    ap.add_argument("--swiglu-dff", type=int, default=256)
    ap.add_argument("--dend-dff", type=int, default=384)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--kwta-frac", type=float, default=0.25)
    ap.add_argument("--dend-init", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()
    args.device = pick_device(args.device)

    data, vocab = build_corpora(args.n_chars, args.seed)
    n_dom = len(data)
    print(f"\nDevice: {args.device}. nanoGPT d_model={args.d_model} x{args.n_layers}L, "
          f"vocab={vocab}, domains={list(data)} (sequential, no replay), "
          f"{args.steps} steps/domain.\n")

    swiglu_ffn = lambda: SwiGLUFFN(args.d_model, args.swiglu_dff)
    dend_ffn = lambda: DendriticGatedFFN(
        args.d_model, args.dend_dff, n_dom, n_segments=args.segments,
        kwta_frac=args.kwta_frac, dend_init=args.dend_init)
    plain_attn = lambda: Attention(args.d_model, args.n_heads)
    gated_attn = lambda: DendriticGatedAttention(
        args.d_model, args.n_heads, n_dom, n_segments=args.segments,
        dend_init=args.dend_init)

    # (ffn, attn): baseline -> gate FFN only -> gate FFN + attention.
    specs = {
        "swiglu":         (swiglu_ffn, plain_attn),
        "dendritic_ffn":  (dend_ffn,   plain_attn),
        "dendritic_full": (dend_ffn,   gated_attn),
    }

    print(f"{'model':16s}{'params':>10s}{'final_mean_bpc':>16s}{'forgetting_bpc':>16s}   per-domain final bpc")
    for name, (mffn, mattn) in specs.items():
        p, per, mean, forg = run(name, mffn, mattn, data, vocab, args, args.device)
        per_s = " ".join(f"{k}={v:.2f}" for k, v in per.items())
        print(f"{name:16s}{p:>10d}{mean:>16.3f}{forg:>16.3f}   {per_s}", flush=True)
    print()


if __name__ == "__main__":
    main()
