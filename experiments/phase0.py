"""Phase 0: validity-first continual-LM comparison on real multi-domain text.

Trains a small GPT SEQUENTIALLY over real domains (wiki / tiny-stories / code,
BPE-tokenized), no replay unless the strategy adds it, and reports proper
continual-learning metrics (avg next-token accuracy, backward transfer =
forgetting, forward transfer) computed from a full R-matrix.

Strategies compared:
  naive            : vanilla SwiGLU GPT, sequential (lower bound).
  replay           : vanilla + rehearsal buffer of past domains.
  lora_per_task    : base trained on domain 0, frozen, per-domain LoRA adapters
                     (parameter isolation; uses task id at test).
  dendritic_oracle : context-routed dendritic FFN+attention, one-hot task id.
  dendritic_free   : SAME, but context is INFERRED from the input (task-free)
                     -> the realistic test: does the benefit survive without
                     being handed the task id at inference?

Usage:
    uv run --no-sync python -u experiments/phase0.py --device cuda
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

from src.blocks import DendriticGatedFFN, DendriticGatedAttention
from src.cl_metrics import cl_metrics
from src.counting import count_params
from src.phase0_data import load_domains
from src.train import set_seed, pick_device


# ------------------------------- LoRA ----------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, n_tasks, r=8):
        super().__init__()
        self.base = nn.Linear(in_f, out_f)
        self.A = nn.Parameter(torch.randn(n_tasks, r, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(n_tasks, out_f, r))
        self.active, self.use_lora = 0, False

    def forward(self, x):
        y = self.base(x)
        if self.use_lora:
            y = y + (x @ self.A[self.active].t()) @ self.B[self.active].t()
        return y


def set_lora(model, active=None, use_lora=None):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if active is not None:
                m.active = active
            if use_lora is not None:
                m.use_lora = use_lora


def split_lora_params(model):
    adapter, base = [], []
    for n, p in model.named_parameters():
        (adapter if (".A" in n or ".B" in n) else base).append(p)
    return adapter, base


# ----------------------------- vanilla blocks --------------------------------
class PlainAttention(nn.Module):
    def __init__(self, d_model, n_heads, lin=nn.Linear):
        super().__init__()
        self.h = n_heads
        self.qkv = lin(d_model, 3 * d_model)
        self.proj = lin(d_model, d_model)

    def forward(self, x, ctx=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        s = (B, T, self.h, C // self.h)
        q, k, v = (t.view(*s).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model, d_ff, lin=nn.Linear):
        super().__init__()
        self.up, self.gate, self.down = lin(d_model, d_ff), lin(d_model, d_ff), lin(d_ff, d_model)

    def forward(self, x, ctx=None):
        return self.down(self.up(x) * F.silu(self.gate(x)))


class ContextNet(nn.Module):
    """Task-free context: infer a soft routing vector from the input itself."""

    def __init__(self, d_model, n_ctx):
        super().__init__()
        self.fc = nn.Linear(d_model, n_ctx)

    def forward(self, x):
        return F.softmax(self.fc(x.mean(dim=1)), dim=-1)


# ------------------------------- model ---------------------------------------
class Block(nn.Module):
    def __init__(self, d_model, attn, ffn):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn, self.ffn = attn, ffn

    def forward(self, x, ctx):
        x = x + self.attn(self.n1(x), ctx)
        x = x + self.ffn(self.n2(x), ctx)
        return x


class GPT(nn.Module):
    def __init__(self, vocab, d_model, n_layers, n_heads, block_size,
                 build_attn, build_ffn, context_mode="none", n_ctx=1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, build_attn(), build_ffn())
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.context_mode = context_mode
        self.ctxnet = ContextNet(d_model, n_ctx) if context_mode == "inferred" else None

    def forward(self, idx, task_ctx=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        if self.context_mode == "oracle":
            ctx = task_ctx
        elif self.context_mode == "inferred":
            ctx = self.ctxnet(x)
        else:
            ctx = None
        for b in self.blocks:
            x = b(x, ctx)
        return F.linear(self.norm(x), self.tok.weight)   # tied head


# ------------------------------ data utils -----------------------------------
def get_batch(data, block, bs, device):
    ix = torch.randint(len(data) - block - 1, (bs,))
    x = torch.stack([data[i:i + block] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix]).to(device)
    return x, y


def onehot(j, n, rows, device):
    v = torch.zeros(rows, n, device=device); v[:, j] = 1.0; return v


@torch.no_grad()
def eval_acc(model, val, ctx_fn, block, bs, device, iters=40):
    model.eval()
    correct = total = 0
    for _ in range(iters):
        x, y = get_batch(val, block, bs, device)
        logits = model(x, ctx_fn(len(x)))
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / total


# ---------------------------- training loops ---------------------------------
def ctx_fn_for(mode, j, n_dom, device):
    if mode == "oracle":
        return lambda rows: onehot(j, n_dom, rows, device)
    return lambda rows: None


def build_R(model, val_list, mode, n_dom, block, bs, device):
    return np.array([eval_acc(model, val_list[j], ctx_fn_for(mode, j, n_dom, device),
                              block, bs, device) for j in range(n_dom)])


def run_standard(model, train_list, val_list, mode, args, device, replay=False):
    n_dom = len(train_list)
    base = build_R(model, val_list, mode, n_dom, args.block, args.batch, device)
    R = np.zeros((n_dom, n_dom))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    buffer = []
    for i in range(n_dom):
        model.train()
        for _ in range(args.steps):
            x, y = get_batch(train_list[i], args.block, args.batch, device)
            ctx = onehot(i, n_dom, len(x), device) if mode == "oracle" else None
            if replay and buffer:
                bi = buffer[np.random.randint(len(buffer))]
                xr, yr = get_batch(train_list[bi], args.block, args.batch // 2, device)
                x = torch.cat([x, xr]); y = torch.cat([y, yr])
                ctx = onehot(i, n_dom, len(x), device) if mode == "oracle" else None
            opt.zero_grad()
            F.cross_entropy(model(x, ctx).view(-1, model.tok.num_embeddings),
                            y.view(-1)).backward()
            opt.step()
        buffer.append(i)
        R[i] = build_R(model, val_list, mode, n_dom, args.block, args.batch, device)
    return base, R


def run_lora(model, train_list, val_list, args, device):
    n_dom = len(train_list)
    base = build_R(model, val_list, "none", n_dom, args.block, args.batch, device)
    R = np.zeros((n_dom, n_dom))
    adapter, base_p = split_lora_params(model)

    # Stage 0: train the full base on domain 0 (LoRA off).
    set_lora(model, use_lora=False)
    opt = torch.optim.AdamW(base_p, lr=args.lr, weight_decay=1e-4)
    model.train()
    for _ in range(args.steps):
        x, y = get_batch(train_list[0], args.block, args.batch, device)
        opt.zero_grad()
        F.cross_entropy(model(x).view(-1, model.tok.num_embeddings), y.view(-1)).backward()
        opt.step()
    # Freeze base; enable per-task adapters.
    for p in base_p:
        p.requires_grad_(False)
    set_lora(model, use_lora=True)

    def eval_row():
        row = np.zeros(n_dom)
        for j in range(n_dom):
            set_lora(model, active=j)
            row[j] = eval_acc(model, val_list[j], lambda r: None, args.block, args.batch, device)
        return row

    R[0] = eval_row()
    for i in range(1, n_dom):
        set_lora(model, active=i)
        opt = torch.optim.AdamW(adapter, lr=args.lr, weight_decay=0.0)
        model.train()
        for _ in range(args.steps):
            x, y = get_batch(train_list[i], args.block, args.batch, device)
            opt.zero_grad()
            F.cross_entropy(model(x).view(-1, model.tok.num_embeddings), y.view(-1)).backward()
            opt.step()
        R[i] = eval_row()
    return base, R


# ------------------------------- build models --------------------------------
def make_model(kind, vocab, args, n_dom):
    d, H, L, nh, bs = args.d_model, args.swiglu_dff, args.n_layers, args.n_heads, args.block
    dff = args.dend_dff
    if kind in ("naive", "replay"):
        return GPT(vocab, d, L, nh, bs,
                   lambda: PlainAttention(d, nh), lambda: SwiGLU_FFN(d, H), "none")
    if kind == "lora":
        lin = lambda i, o: LoRALinear(i, o, n_dom, args.lora_r)
        return GPT(vocab, d, L, nh, bs,
                   lambda: PlainAttention(d, nh, lin), lambda: SwiGLU_FFN(d, H, lin), "none")
    if kind in ("dend_oracle", "dend_free"):
        mode = "oracle" if kind == "dend_oracle" else "inferred"
        n_ctx = n_dom if mode == "oracle" else args.ctx_dim
        return GPT(vocab, d, L, nh, bs,
                   lambda: DendriticGatedAttention(d, nh, n_ctx, n_segments=args.segments,
                                                   dend_init=args.dend_init),
                   lambda: DendriticGatedFFN(d, dff, n_ctx, n_segments=args.segments,
                                             kwta_frac=args.kwta_frac, dend_init=args.dend_init),
                   mode, n_ctx)
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
    ap.add_argument("--dend-dff", type=int, default=768)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--kwta-frac", type=float, default=0.25)
    ap.add_argument("--dend-init", type=float, default=1.5)
    ap.add_argument("--ctx-dim", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--max-chars", type=int, default=1_500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()
    args.device = pick_device(args.device)
    device = args.device

    print(f"\nDevice: {device}. Loading domains...")
    data, vocab, kind = load_domains(max_chars=args.max_chars)
    names = list(data)
    train_list = [torch.from_numpy(data[n][0]) for n in names]
    val_list = [torch.from_numpy(data[n][1]) for n in names]
    n_dom = len(names)
    print(f"\nPhase 0: domains={names} (sequential, no replay unless noted), "
          f"vocab={vocab}, d_model={args.d_model}x{args.n_layers}L, steps={args.steps}/domain.\n")

    strategies = ["naive", "replay", "lora", "dend_oracle", "dend_free"]
    print(f"{'strategy':16s}{'params':>11s}{'avg_acc':>9s}{'BWT':>8s}{'FWT':>8s}{'forget':>8s}   "
          f"task-id@test?")
    for s in strategies:
        set_seed(args.seed)
        model = make_model(s, vocab, args, n_dom).to(device)
        if s == "replay":
            base, R = run_standard(model, train_list, val_list, "none", args, device, replay=True)
        elif s == "lora":
            base, R = run_lora(model, train_list, val_list, args, device)
        else:
            mode = "oracle" if s == "dend_oracle" else ("inferred" if s == "dend_free" else "none")
            base, R = run_standard(model, train_list, val_list, mode, args, device)
        m = cl_metrics(R, base)
        needs_id = "yes" if s in ("lora", "dend_oracle") else "no"
        print(f"{s:16s}{count_params(model):>11d}{m['avg_acc']*100:>8.1f}%"
              f"{m['bwt']*100:>7.1f}%{m['fwt']*100:>7.1f}%{m['forgetting']*100:>7.1f}%   {needs_id}",
              flush=True)
    print("\n(avg_acc/BWT/FWT/forget are next-token top-1 accuracy; "
          "BWT<0 and forget>0 mean catastrophic forgetting.)\n")


if __name__ == "__main__":
    main()
