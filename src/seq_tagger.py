"""Shared per-position sequence-tagger scaffold for the state-tracking probes.

Both streaming experiments (parity, mod-k counter) use the same model: a stack
of pre-norm (token-mixer + SwiGLU FFN) blocks with a per-position head, and only
the TOKEN MIXER is swapped at a matched per-mixer parameter budget. The mixer
zoo and the sizing logic live here so the experiment scripts only own their task,
loss, and reporting.

  attention      causal self-attention (with learned positions)
  mamba          flat selective SSM (linear recurrence)
  ssm_coinc      two selective-memory streams multiplied + plateau
  dendritic_ssm  tree of plateau-gated selective branches + multiplicative soma
  dendritic_rec  selective SSM with the regenerative + signed gate IN the loop
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.blocks import SwiGLU
from src.counting import size_to_budget
from src.ssm import (CoincidenceSSM, DendriticSSMBlock, MambaBlock,
                     RecurrentDendriticBlock, RotationRecurrentBlock)


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
    "dendritic_rec": MixerSpec(
        lambda w, c: RecurrentDendriticBlock(c.d_model, d_inner=_w(w), d_state=c.d_state,
                                             conv_k=c.conv_k), False),
    "dendritic_rot": MixerSpec(
        lambda w, c: RotationRecurrentBlock(c.d_model, d_inner=_w(w), d_state=c.d_state,
                                            conv_k=c.conv_k), False),
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


class SeqTagger(nn.Module):
    """Bit-stream -> per-position logits.

    ``n_out=1`` returns ``(B, T)`` logits for a binary (BCE) target; ``n_out>1``
    returns ``(B, T, n_out)`` class logits for a multi-class (cross-entropy)
    target. The input alphabet is binary in both cases (a {0,1} stream).
    """

    def __init__(self, d_model: int, n_layers: int, make_mixer: Callable[[], nn.Module],
                 ffn_hidden: int, max_len: int, use_pos: bool, n_out: int = 1):
        super().__init__()
        self.n_out = n_out
        self.emb = nn.Embedding(2, d_model)
        self.pos = nn.Embedding(max_len, d_model) if use_pos else None
        self.blocks = nn.ModuleList(
            Block(d_model, make_mixer, ffn_hidden) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_out)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        x = self.emb(bits)
        if self.pos is not None:
            T = bits.shape[1]
            if T > self.pos.num_embeddings:
                raise ValueError(f"seq_len {T} exceeds max_len {self.pos.num_embeddings}")
            x = x + self.pos(torch.arange(T, device=bits.device))[None]
        for b in self.blocks:
            x = b(x)
        out = self.head(self.norm(x))                     # (B, T, n_out)
        return out.squeeze(-1) if self.n_out == 1 else out
