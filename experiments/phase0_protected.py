"""Phase 0b: can PROTECTING the shared bulk make the dendritic transformer
competitive with replay on continual multi-domain LM?

Phase 0 showed dendritic FFN/attention gating still forgets badly because the
gate only touches a thin slice while embeddings, head, QKV and norms are shared
and overwritten. Here we test the fix: make the shared bulk context-conditioned
too, and modernize the transformer so every model is a strong, fair baseline.

Modern arch (ALL models): RMSNorm, RoPE, no learned pos-emb, AdamW with
warmup+cosine LR and gradient clipping.

Context conditioning (dendritic models only), routed by a context vector that
is either an oracle one-hot task id or INFERRED from the input (task-free):
  - dendritic gated FFN + attention      (as in Phase 0)
  - per-domain embedding FiLM (scale/shift)         <- protects tied emb/head
  - conditional RMSNorm gains                         <- protects normalization

Strategies:
  naive+        : modern arch, no context (lower bound)
  replay+       : modern arch + rehearsal buffer (target to beat)
  dend_gate+    : dendritic FFN/attn gating only, oracle ctx (Phase-0 method, modern arch)
  dend_protect+ : gating + embedding FiLM + conditional norm, oracle ctx
  dend_free+    : same protection but TASK-FREE inferred ctx (realistic)

Usage:
    uv run --no-sync python -u experiments/phase0_protected.py --device cuda
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

from src.blocks import kwta
from src.cl_metrics import cl_metrics
from src.counting import count_params
from src.phase0_data import load_domains
from src.train import set_seed, pick_device


# ------------------------------- RoPE ----------------------------------------
def rope_tables(T, hd, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    freqs = torch.outer(torch.arange(T, device=device).float(), inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):  # x: (B, h, T, hd)
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


# ------------------------------- norms ---------------------------------------
class CondRMSNorm(nn.Module):
    """RMSNorm whose gain can be shifted per context (conditional norm)."""

    def __init__(self, d, n_ctx=0):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.cond = nn.Parameter(torch.zeros(n_ctx, d)) if n_ctx else None

    def forward(self, x, ctx=None):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        if self.cond is not None and ctx is not None:
            return x * (self.g + ctx @ self.cond).unsqueeze(1)
        return x * self.g


# ------------------------------- layers --------------------------------------
class Attn(nn.Module):
    def __init__(self, d, h, n_ctx=0, n_seg=8, dend_init=1.5, gate=False):
        super().__init__()
        self.h, self.hd = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.dend = nn.Parameter(torch.randn(d, n_seg, n_ctx) * dend_init) if gate else None

    def forward(self, x, ctx, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        sh = (B, T, self.h, self.hd)
        q, k, v = (t.view(*sh).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        if self.dend is not None and ctx is not None:
            a = torch.einsum("bc,fsc->bfs", ctx, self.dend)
            w = torch.gather(a, -1, a.abs().argmax(-1, keepdim=True)).squeeze(-1)
            y = y * torch.sigmoid(w).unsqueeze(1)
        return self.proj(y)


class FFN(nn.Module):
    """SwiGLU FFN; optional context-routed dendritic gate + kWTA."""

    def __init__(self, d, dff, n_ctx=0, n_seg=8, kwta_frac=0.25, dend_init=1.5, gate=False):
        super().__init__()
        self.up, self.gate, self.down = nn.Linear(d, dff), nn.Linear(d, dff), nn.Linear(dff, d)
        self.frac = kwta_frac
        self.dend = nn.Parameter(torch.randn(dff, n_seg, n_ctx) * dend_init) if gate else None

    def forward(self, x, ctx):
        h = self.up(x) * F.silu(self.gate(x))
        if self.dend is not None and ctx is not None:
            a = torch.einsum("bc,fsc->bfs", ctx, self.dend)
            w = torch.gather(a, -1, a.abs().argmax(-1, keepdim=True)).squeeze(-1)
            h = kwta(h * torch.sigmoid(w).unsqueeze(1), self.frac)
        return self.down(h)


class ContextNet(nn.Module):
    def __init__(self, d, n_ctx):
        super().__init__()
        self.fc = nn.Linear(d, n_ctx)

    def forward(self, x):
        return F.softmax(self.fc(x.mean(dim=1)), dim=-1)


class Block(nn.Module):
    def __init__(self, d, h, n_ctx_cond, gate, args):
        super().__init__()
        self.n1 = CondRMSNorm(d, n_ctx_cond)
        self.n2 = CondRMSNorm(d, n_ctx_cond)
        self.attn = Attn(d, h, args._n_ctx, args.segments, args.dend_init, gate)
        self.ffn = FFN(d, args.dend_dff if gate else args.swiglu_dff,
                       args._n_ctx, args.segments, args.kwta_frac, args.dend_init, gate)

    def forward(self, x, ctx, cos, sin):
        x = x + self.attn(self.n1(x, ctx), ctx, cos, sin)
        x = x + self.ffn(self.n2(x, ctx), ctx)
        return x


class GPT(nn.Module):
    def __init__(self, vocab, args, context_mode="none", gate=False,
                 film=False, cond_norm=False):
        super().__init__()
        d = args.d_model
        self.context_mode = context_mode
        self._n_ctx = args._n_ctx
        n_ctx_cond = args._n_ctx if cond_norm else 0
        self.tok = nn.Embedding(vocab, d)
        self.ctxnet = ContextNet(d, args._n_ctx) if context_mode == "inferred" else None
        if film:
            self.film_g = nn.Parameter(torch.zeros(args._n_ctx, d))
            self.film_b = nn.Parameter(torch.zeros(args._n_ctx, d))
        else:
            self.film_g = None
        self.blocks = nn.ModuleList([Block(d, args.n_heads, n_ctx_cond, gate, args)
                                     for _ in range(args.n_layers)])
        self.norm = CondRMSNorm(d, n_ctx_cond)
        self.hd = d // args.n_heads

    def forward(self, idx, task_ctx=None):
        B, T = idx.shape
        x = self.tok(idx)
        if self.context_mode == "oracle":
            ctx = task_ctx
        elif self.context_mode == "inferred":
            ctx = self.ctxnet(x)
        else:
            ctx = None
        if self.film_g is not None and ctx is not None:
            x = x * (1 + (ctx @ self.film_g).unsqueeze(1)) + (ctx @ self.film_b).unsqueeze(1)
        cos, sin = rope_tables(T, self.hd, idx.device)
        for b in self.blocks:
            x = b(x, ctx, cos, sin)
        return F.linear(self.norm(x, ctx), self.tok.weight)


# ------------------------------ data / eval ----------------------------------
def get_batch(data, block, bs, device):
    ix = torch.randint(len(data) - block - 1, (bs,))
    x = torch.stack([data[i:i + block] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix]).to(device)
    return x, y


def onehot(j, n, rows, device):
    v = torch.zeros(rows, n, device=device); v[:, j] = 1.0; return v


def ctx_fn_for(mode, j, n_dom, device):
    if mode == "oracle":
        return lambda rows: onehot(j, n_dom, rows, device)
    return lambda rows: None


@torch.no_grad()
def eval_acc(model, val, ctx_fn, block, bs, device, iters=30):
    model.eval()
    correct = total = 0
    for _ in range(iters):
        x, y = get_batch(val, block, bs, device)
        logits = model(x, ctx_fn(len(x)))
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / total


def build_R(model, val_list, mode, n_dom, block, bs, device):
    return np.array([eval_acc(model, val_list[j], ctx_fn_for(mode, j, n_dom, device),
                              block, bs, device) for j in range(n_dom)])


def lr_at(step, total, base, warm=0.1, lo=0.1):
    w = max(1, int(total * warm))
    if step < w:
        return base * step / w
    p = (step - w) / max(1, total - w)
    return base * (lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * p)))


def run_standard(model, train_list, val_list, mode, args, device, replay=False):
    n_dom = len(train_list)
    base = build_R(model, val_list, mode, n_dom, args.block, args.batch, device)
    R = np.zeros((n_dom, n_dom))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    V = model.tok.num_embeddings
    buffer = []
    for i in range(n_dom):
        model.train()
        for s in range(args.steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(s, args.steps, args.lr)
            x, y = get_batch(train_list[i], args.block, args.batch, device)
            if replay and buffer:
                bi = buffer[np.random.randint(len(buffer))]
                xr, yr = get_batch(train_list[bi], args.block, args.batch // 2, device)
                x, y = torch.cat([x, xr]), torch.cat([y, yr])
            ctx = onehot(i, n_dom, len(x), device) if mode == "oracle" else None
            opt.zero_grad()
            F.cross_entropy(model(x, ctx).view(-1, V), y.view(-1)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        buffer.append(i)
        R[i] = build_R(model, val_list, mode, n_dom, args.block, args.batch, device)
    return base, R


def make_model(kind, vocab, args, n_dom):
    if kind in ("naive", "replay"):
        args._n_ctx = 1
        return GPT(vocab, args, "none", gate=False)
    if kind == "dend_gate":
        args._n_ctx = n_dom
        return GPT(vocab, args, "oracle", gate=True, film=False, cond_norm=False)
    if kind == "dend_protect":
        args._n_ctx = n_dom
        return GPT(vocab, args, "oracle", gate=True, film=True, cond_norm=True)
    if kind == "dend_free":
        args._n_ctx = args.ctx_dim
        return GPT(vocab, args, "inferred", gate=True, film=True, cond_norm=True)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--swiglu-dff", type=int, default=512)
    ap.add_argument("--dend-dff", type=int, default=512)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--kwta-frac", type=float, default=0.25)
    ap.add_argument("--dend-init", type=float, default=1.5)
    ap.add_argument("--ctx-dim", type=int, default=16)
    ap.add_argument("--max-chars", type=int, default=1_500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()
    args.device = pick_device(args.device)
    args._n_ctx = 1
    device = args.device

    print(f"\nDevice: {device}. Loading domains...")
    data, vocab, kind = load_domains(max_chars=args.max_chars)
    names = list(data)
    train_list = [torch.from_numpy(data[n][0]) for n in names]
    val_list = [torch.from_numpy(data[n][1]) for n in names]
    n_dom = len(names)
    print(f"\nPhase 0b (modern arch: RMSNorm+RoPE+warmup/cosine+clip): domains={names}, "
          f"vocab={vocab}, d_model={args.d_model}x{args.n_layers}L, steps={args.steps}/domain.\n")

    strategies = ["naive", "replay", "dend_gate", "dend_protect", "dend_free"]
    print(f"{'strategy':16s}{'params':>11s}{'avg_acc':>9s}{'BWT':>8s}{'FWT':>8s}{'forget':>8s}   task-id@test?")
    for s in strategies:
        set_seed(args.seed)
        model = make_model(s, vocab, args, n_dom).to(device)
        mode = "oracle" if s in ("dend_gate", "dend_protect") else ("inferred" if s == "dend_free" else "none")
        base, R = run_standard(model, train_list, val_list, mode, args, device,
                               replay=(s == "replay"))
        m = cl_metrics(R, base)
        needs_id = "yes" if s in ("dend_gate", "dend_protect") else "no"
        print(f"{s:16s}{count_params(model):>11d}{m['avg_acc']*100:>8.1f}%"
              f"{m['bwt']*100:>7.1f}%{m['fwt']*100:>7.1f}%{m['forgetting']*100:>7.1f}%   {needs_id}",
              flush=True)
    print("\n(BWT<0 and forget>0 mean catastrophic forgetting; replay+ is the target to beat.)\n")


if __name__ == "__main__":
    main()
