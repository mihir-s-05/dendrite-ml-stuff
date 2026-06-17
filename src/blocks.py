"""FFN-style blocks for the dendritic-vs-point-neuron comparison.

Each block maps d_model -> d_model so it is a drop-in replacement for a
transformer feed-forward sub-layer. We compare three families:

  - MLP     : the standard point-neuron FFN (Linear -> act -> Linear).
  - SwiGLU  : the gated FFN used in modern LLMs (a multiplicative baseline).
  - Dendritic: a block inspired by "What can a neuron compute?" (Aizenbud
              et al. 2026). It groups hidden units into K dendritic *branches*
              (subunits), applies a local nonlinearity with an NMDA-like
              multiplicative gate inside each branch, and integrates branch
              outputs at a "soma" with an optional high-order cross-branch
              interaction (the analogue of a dendritic Ca2+ spike combining
              subunit outputs).

The scientific question is not "is the dendritic block more powerful?" (with
more params it trivially is) but "is the dendritic structure a better
inductive bias at MATCHED parameter budget?". All sizing is done by param
budget in counting.py so the comparison is fair.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Standard point-neuron FFN: sum -> nonlinearity -> sum."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class SwiGLU(nn.Module):
    """Gated FFN (SwiGLU). A strong multiplicative baseline."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(d_model, hidden)
        self.gate = nn.Linear(d_model, hidden)
        self.down = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.up(x) * F.silu(self.gate(x)))


class DendriticFFN(nn.Module):
    """Dendritic block: branches with local gating + soma integration.

    Args:
        d_model: model width (input and output dim).
        n_branches: number of dendritic subunits (K).
        branch_dim: hidden units per branch (so hidden = K * branch_dim).
        nmda_gate: enable the NMDA-like multiplicative gate inside branches.
        branch_interaction: enable the high-order cross-branch ("soma Ca2+
            spike") quadratic term that binds subunit outputs.
        local_input: if True, each branch reads only its own slice of a
            grouped input projection (structured sparsity / locality), which
            cuts parameters; if False, every branch reads the full input.
    """

    def __init__(
        self,
        d_model: int,
        n_branches: int = 16,
        branch_dim: int = 8,
        nmda_gate: bool = True,
        branch_interaction: bool = True,
        local_input: bool = False,
    ):
        super().__init__()
        self.K = n_branches
        self.b = branch_dim
        self.hidden = n_branches * branch_dim
        self.nmda_gate = nmda_gate
        self.branch_interaction = branch_interaction
        self.local_input = local_input

        if local_input:
            # Branch k reads a disjoint slice of the input (structured sparsity
            # / sub-unit locality). The input is zero-padded so any branch
            # count works without a divisibility constraint.
            self.slice = math.ceil(d_model / n_branches)
            self.pad = self.slice * n_branches - d_model
            self.in_w = nn.Parameter(torch.empty(n_branches, self.slice, branch_dim))
            self.in_b = nn.Parameter(torch.zeros(n_branches, branch_dim))
            nn.init.kaiming_uniform_(self.in_w, a=5 ** 0.5)
            if nmda_gate:
                self.gate_w = nn.Parameter(torch.empty(n_branches, self.slice, branch_dim))
                self.gate_b = nn.Parameter(torch.zeros(n_branches, branch_dim))
                nn.init.kaiming_uniform_(self.gate_w, a=5 ** 0.5)
        else:
            self.in_proj = nn.Linear(d_model, self.hidden)
            if nmda_gate:
                self.gate_proj = nn.Linear(d_model, self.hidden)

        self.soma = nn.Linear(self.hidden, d_model)

        if branch_interaction:
            # Low-order mixing of per-branch summaries, then a multiplicative
            # (quadratic) term: the dendritic Ca2+-spike-like binding of
            # subunit outputs. Cheap: O(K^2) params.
            self.inter_in = nn.Linear(n_branches, n_branches)
            self.inter_out = nn.Linear(n_branches, d_model)

    def _branch_preact(self, x: torch.Tensor):
        if self.local_input:
            if self.pad:
                x = F.pad(x, (0, self.pad))
            *lead, d = x.shape
            xg = x.view(*lead, self.K, self.slice)
            pre = torch.einsum("...ks,ksb->...kb", xg, self.in_w) + self.in_b
            if self.nmda_gate:
                gate = torch.einsum("...ks,ksb->...kb", xg, self.gate_w) + self.gate_b
            else:
                gate = None
        else:
            *lead, d = x.shape
            pre = self.in_proj(x).view(*lead, self.K, self.b)
            gate = self.gate_proj(x).view(*lead, self.K, self.b) if self.nmda_gate else None
        return pre, gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre, gate = self._branch_preact(x)
        a = F.gelu(pre)
        if gate is not None:
            a = a * torch.sigmoid(gate)  # NMDA-like coincidence gate

        *lead, K, b = a.shape
        out = self.soma(a.reshape(*lead, K * b))

        if self.branch_interaction:
            bsum = a.sum(dim=-1)  # per-branch output, shape (..., K)
            mixed = self.inter_in(bsum)
            high_order = bsum * mixed  # quadratic cross-branch binding
            out = out + self.inter_out(high_order)
        return out


class TemporalDendriticFFN(nn.Module):
    """Transformer FFN replacement with dendritic routing and temporal memory.

    This is the next-step hypothesis from the TwinProp paper: keep the block
    transformer-compatible, but restore three missing ingredients from the
    static `DendriticFFN` probe:

      - learned branch-specific input routing (a proxy for synaptic location),
      - causal branch-local memory (a proxy for slow dendritic/synaptic state),
      - NMDA-like coincidence and soma-level branch binding.

    Input/output shape is `(batch, time, d_model)`, matching a standard FFN
    sub-layer. A 2D `(batch, d_model)` input is treated as a length-1 sequence
    so the block remains easy to smoke-test.
    """

    def __init__(
        self,
        d_model: int,
        n_branches: int = 16,
        branch_dim: int = 8,
        memory_kernel: int = 9,
        nmda_gate: bool = True,
        branch_interaction: bool = True,
        routed_input: bool = True,
    ):
        super().__init__()
        if memory_kernel < 1:
            raise ValueError("memory_kernel must be >= 1")
        self.K = n_branches
        self.b = branch_dim
        self.hidden = n_branches * branch_dim
        self.memory_kernel = memory_kernel
        self.nmda_gate = nmda_gate
        self.branch_interaction = branch_interaction
        self.routed_input = routed_input

        if routed_input:
            self.route_logits = nn.Parameter(torch.zeros(n_branches, d_model))
            self.in_w = nn.Parameter(torch.empty(n_branches, d_model, branch_dim))
            self.in_b = nn.Parameter(torch.zeros(n_branches, branch_dim))
            nn.init.kaiming_uniform_(self.in_w, a=5 ** 0.5)
            if nmda_gate:
                self.gate_w = nn.Parameter(torch.empty(n_branches, d_model, branch_dim))
                self.gate_b = nn.Parameter(torch.zeros(n_branches, branch_dim))
                nn.init.kaiming_uniform_(self.gate_w, a=5 ** 0.5)
        else:
            self.in_proj = nn.Linear(d_model, self.hidden)
            if nmda_gate:
                self.gate_proj = nn.Linear(d_model, self.hidden)

        self.memory = nn.Conv1d(
            self.hidden,
            self.hidden,
            kernel_size=memory_kernel,
            groups=self.hidden,
            bias=False,
        )
        self.gate_memory = (
            nn.Conv1d(self.hidden, self.hidden, kernel_size=memory_kernel, groups=self.hidden, bias=False)
            if nmda_gate
            else None
        )
        self.soma = nn.Linear(self.hidden, d_model)

        if branch_interaction:
            self.inter_in = nn.Linear(n_branches, n_branches)
            self.inter_out = nn.Linear(n_branches, d_model)

        self.reset_memory_parameters()

    def reset_memory_parameters(self) -> None:
        """Initialize causal memory as mostly-present plus decaying history."""
        with torch.no_grad():
            taps = torch.arange(self.memory_kernel, dtype=self.memory.weight.dtype)
            # Conv1d sees left-padded history in chronological order; the last
            # tap is the current token, earlier taps are exponentially smaller.
            decay = torch.exp(-(self.memory_kernel - 1 - taps) / max(1.0, self.memory_kernel / 3.0))
            decay = decay / decay.sum()
            self.memory.weight.copy_(decay.view(1, 1, -1).repeat(self.hidden, 1, 1))
            if self.gate_memory is not None:
                self.gate_memory.weight.copy_(decay.view(1, 1, -1).repeat(self.hidden, 1, 1))

    def _branch_preact(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.routed_input:
            route = torch.softmax(self.route_logits, dim=-1) * x.shape[-1]
            routed = x.unsqueeze(-2) * route
            pre = torch.einsum("...kd,kdb->...kb", routed, self.in_w) + self.in_b
            if self.nmda_gate:
                gate = torch.einsum("...kd,kdb->...kb", routed, self.gate_w) + self.gate_b
            else:
                gate = None
        else:
            *lead, _ = x.shape
            pre = self.in_proj(x).view(*lead, self.K, self.b)
            gate = self.gate_proj(x).view(*lead, self.K, self.b) if self.nmda_gate else None
        return pre, gate

    def _causal_memory(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        B, T, K, b = x.shape
        flat = x.reshape(B, T, K * b).transpose(1, 2)
        flat = F.pad(flat, (self.memory_kernel - 1, 0))
        return conv(flat).transpose(1, 2).reshape(B, T, K, b)

    def routing_entropy(self) -> torch.Tensor:
        """Mean normalized entropy of branch routing masks; lower means sharper routing."""
        if not self.routed_input:
            return torch.tensor(1.0, device=self.soma.weight.device)
        probs = torch.softmax(self.route_logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        return (entropy / math.log(probs.shape[-1])).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_time = x.dim() == 2
        if squeeze_time:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError("TemporalDendriticFFN expects (B, T, C) or (B, C)")

        pre, gate = self._branch_preact(x)
        pre = self._causal_memory(pre, self.memory)
        a = F.gelu(pre)
        if gate is not None and self.gate_memory is not None:
            gate = self._causal_memory(gate, self.gate_memory)
            a = a * torch.sigmoid(gate)

        B, T, K, b = a.shape
        out = self.soma(a.reshape(B, T, K * b))
        if self.branch_interaction:
            bsum = a.sum(dim=-1)
            high_order = bsum * self.inter_in(bsum)
            out = out + self.inter_out(high_order)
        return out.squeeze(1) if squeeze_time else out


def kwta(h: torch.Tensor, frac: float) -> torch.Tensor:
    """k-Winner-Take-All: keep the top `frac` activations per row, zero the rest.

    Sparse, low-overlap codes are what let context-routed dendrites avoid
    catastrophic forgetting (Iyer et al. 2022); without this, shared weights
    still get overwritten across tasks.
    """
    H = h.shape[-1]
    k = max(1, int(round(frac * H)))
    if k >= H:
        return h
    thresh = torch.topk(h, k, dim=-1).values[..., -1:]
    return h * (h >= thresh)


class KWTAMLP(nn.Module):
    """Control: sparse (kWTA) MLP that ignores context. Isolates whether
    sparsity ALONE (without dendritic routing) prevents forgetting."""

    def __init__(self, d_in: int, n_ctx: int, d_hidden: int, n_out: int = 1,
                 kwta_frac: float = 0.2, **_):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, n_out)
        self.frac = kwta_frac

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.fc2(kwta(F.relu(self.fc1(x)), self.frac))


class ContextMLP(nn.Module):
    """Baseline that IGNORES the task context (expected to catastrophically
    forget under sequential multi-task training)."""

    def __init__(self, d_in: int, n_ctx: int, d_hidden: int, n_out: int = 1, **_):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, n_out)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class ConcatContextMLP(nn.Module):
    """Baseline that concatenates the one-hot context to the input."""

    def __init__(self, d_in: int, n_ctx: int, d_hidden: int, n_out: int = 1, **_):
        super().__init__()
        self.fc1 = nn.Linear(d_in + n_ctx, d_hidden)
        self.fc2 = nn.Linear(d_hidden, n_out)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(torch.cat([x, ctx], dim=-1))))


class ActiveDendriteMLP(nn.Module):
    """Active-Dendrites MLP (Iyer et al. 2022).

    Each hidden unit owns several dendritic segments. The task context picks
    (winner-take-all by absolute activation) one segment per unit, whose value
    multiplicatively gates that unit. Context-dependent gating lets different
    tasks recruit different sub-populations -> less catastrophic forgetting.
    """

    def __init__(self, d_in: int, n_ctx: int, d_hidden: int, n_out: int = 1,
                 n_segments: int = 4, kwta_frac: float = 0.2,
                 dend_init: float = 1.0, **_):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        # Init large enough that sigmoid(winner) spans ~[0.1, 0.9] across units
        # from the start, so different tasks route to different sub-populations
        # immediately (weak init -> all gates ~0.5 -> no routing -> no benefit).
        self.dend = nn.Parameter(torch.randn(d_hidden, n_segments, n_ctx) * dend_init)
        self.fc2 = nn.Linear(d_hidden, n_out)
        self.frac = kwta_frac

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        a = torch.einsum("bc,hsc->bhs", ctx, self.dend)  # dendritic activations
        idx = a.abs().argmax(dim=-1, keepdim=True)
        winner = torch.gather(a, -1, idx).squeeze(-1)     # signed winning segment
        h = F.relu(h) * torch.sigmoid(winner)
        h = kwta(h, self.frac)                            # sparse, low-overlap code
        return self.fc2(h)


class DendriticGatedFFN(nn.Module):
    """Transformer FFN sub-layer with context-routed dendritic gating + kWTA.

    A drop-in for the SwiGLU/MLP FFN, but each hidden unit owns several
    dendritic segments; a per-sequence context vector (e.g. task/domain id)
    selects a segment per unit (winner-take-all) whose value multiplicatively
    gates the unit. With kWTA this routes different contexts to different
    sparse sub-populations -> resists catastrophic forgetting across domains.
    """

    def __init__(self, d_model: int, d_ff: int, n_ctx: int, n_segments: int = 8,
                 kwta_frac: float = 0.1, dend_init: float = 1.5):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        self.dend = nn.Parameter(torch.randn(d_ff, n_segments, n_ctx) * dend_init)
        self.frac = kwta_frac

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        h = self.up(x)                                   # (B, T, d_ff)
        a = torch.einsum("bc,fsc->bfs", ctx, self.dend)  # (B, d_ff, n_seg)
        winner = torch.gather(a, -1, a.abs().argmax(-1, keepdim=True)).squeeze(-1)
        gate = torch.sigmoid(winner).unsqueeze(1)        # (B, 1, d_ff) -> over T
        h = kwta(F.gelu(h) * gate, self.frac)
        return self.down(h)


class DendriticGatedAttention(nn.Module):
    """Self-attention whose per-channel output is context-routed (dendritic
    gate) before the output projection, so different contexts/domains use
    different attention subspaces and downstream weights forget less."""

    def __init__(self, d_model: int, n_heads: int, n_ctx: int,
                 n_segments: int = 8, dend_init: float = 1.5):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dend = nn.Parameter(torch.randn(d_model, n_segments, n_ctx) * dend_init)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = (B, T, self.h, C // self.h)
        q, k, v = (t.view(*shape).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return y.transpose(1, 2).contiguous().view(B, T, C)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        y = self._attn(x)
        a = torch.einsum("bc,fsc->bfs", ctx, self.dend)
        winner = torch.gather(a, -1, a.abs().argmax(-1, keepdim=True)).squeeze(-1)
        gate = torch.sigmoid(winner).unsqueeze(1)         # (B, 1, C) over T
        return self.proj(y * gate)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN with a context arg (ignored) for a uniform call signature."""

    def __init__(self, d_model: int, d_ff: int, **_):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.gate = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        return self.down(self.up(x) * F.silu(self.gate(x)))


class Residual(nn.Module):
    """Pre-norm residual wrapper around an FFN block."""

    def __init__(self, d_model: int, block: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(self.norm(x))


class BlockClassifier(nn.Module):
    """Linear embed -> N residual FFN blocks -> linear head.

    With n_layers=1 this isolates a single FFN block's expressivity, matching
    the paper's "what can a single unit compute" framing.
    """

    def __init__(self, d_in: int, d_model: int, make_block, n_layers: int = 1, n_out: int = 1):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([Residual(d_model, make_block()) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


class CausalConvSwiGLU(nn.Module):
    """Non-dendritic temporal baseline: a causal depthwise conv (temporal
    mixing) followed by a plain pointwise SwiGLU.

    This isolates the question the dendritic ablations leave open: is the gain
    on temporal tasks due to *dendritic* structure, or simply due to having
    ANY temporal mixing? It uses the same causal depthwise-conv primitive as
    TemporalDendriticFFN's "branch memory" but with zero dendritic machinery
    (no branches, routing, NMDA gate, or soma interaction).
    """

    def __init__(self, d_model: int, hidden: int, memory_kernel: int = 9):
        super().__init__()
        self.memory_kernel = memory_kernel
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=memory_kernel,
                              groups=d_model, bias=False)
        self.up = nn.Linear(d_model, hidden)
        self.gate = nn.Linear(d_model, hidden)
        self.down = nn.Linear(hidden, d_model)
        self._init_decay_kernel()

    def _init_decay_kernel(self) -> None:
        """Match TemporalDendriticFFN's memory init: a causal exponential-decay
        kernel (recent emphasized) so this is a fair temporal mixer, not a
        randomly-initialized one."""
        with torch.no_grad():
            taps = torch.arange(self.memory_kernel, dtype=self.conv.weight.dtype)
            decay = torch.exp(-(self.memory_kernel - 1 - taps) / max(1.0, self.memory_kernel / 3.0))
            decay = decay / decay.sum()
            self.conv.weight.copy_(decay.view(1, 1, -1).repeat(self.conv.weight.shape[0], 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_time = x.dim() == 2
        if squeeze_time:
            x = x.unsqueeze(1)
        h = x.transpose(1, 2)
        h = F.pad(h, (self.memory_kernel - 1, 0))
        h = self.conv(h).transpose(1, 2)
        out = self.down(self.up(h) * F.silu(self.gate(h)))
        return out.squeeze(1) if squeeze_time else out


class GatedConvFFN(nn.Module):
    """Coincidence control: TemporalDendriticFFN stripped to its core mechanism
    (per-channel temporal memory + NMDA-like multiplicative gate) with NO
    branches, routing, or soma interaction.

    Two pointwise projections each get their own causal depthwise-conv memory,
    then a multiplicative coincidence gate combines them:
        h = gelu(conv_a(up(x))) * sigmoid(conv_b(gate(x)))
    This isolates the question: is the d=6 dendritic advantage from the
    *coincidence-of-temporally-filtered-streams* mechanism (which any mixer
    could adopt), or from the branch/soma structure specifically?
    """

    def __init__(self, d_model: int, hidden: int, memory_kernel: int = 9):
        super().__init__()
        self.memory_kernel = memory_kernel
        self.up = nn.Linear(d_model, hidden)
        self.gate = nn.Linear(d_model, hidden)
        self.down = nn.Linear(hidden, d_model)
        self.conv_a = nn.Conv1d(hidden, hidden, memory_kernel, groups=hidden, bias=False)
        self.conv_b = nn.Conv1d(hidden, hidden, memory_kernel, groups=hidden, bias=False)
        self._init_decay()

    def _init_decay(self) -> None:
        with torch.no_grad():
            taps = torch.arange(self.memory_kernel, dtype=self.conv_a.weight.dtype)
            decay = torch.exp(-(self.memory_kernel - 1 - taps) / max(1.0, self.memory_kernel / 3.0))
            decay = decay / decay.sum()
            for conv in (self.conv_a, self.conv_b):
                conv.weight.copy_(decay.view(1, 1, -1).repeat(conv.weight.shape[0], 1, 1))

    def _mem(self, h: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        h = h.transpose(1, 2)
        h = F.pad(h, (self.memory_kernel - 1, 0))
        return conv(h).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_time = x.dim() == 2
        if squeeze_time:
            x = x.unsqueeze(1)
        a = self._mem(self.up(x), self.conv_a)
        b = self._mem(self.gate(x), self.conv_b)
        out = self.down(F.gelu(a) * torch.sigmoid(b))
        return out.squeeze(1) if squeeze_time else out


class TemporalBlockClassifier(nn.Module):
    """Sequence classifier around transformer-style temporal FFN blocks.

    The input is an event/spike tensor `(batch, time, channels)`. Each timestep
    is embedded to `d_model`, processed by residual FFN blocks, then pooled over
    a decision window to mimic the paper's spike/no-spike readout window.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int,
        make_block,
        n_layers: int = 1,
        n_out: int = 1,
        decision_window: int = 8,
    ):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([Residual(d_model, make_block()) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, n_out)
        self.decision_window = decision_window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        pooled = h[:, -self.decision_window :].mean(dim=1)
        return self.head(pooled)
