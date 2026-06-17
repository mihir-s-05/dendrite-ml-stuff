"""Selective SSM (Mamba-style) baseline + a Dendritic-SSM block.

Goal: test whether genuinely dendritic structure - a TREE of nonlinear
subunits with regenerative PLATEAU states and a multiplicative SOMA - buys
anything over a flat selective SSM (Mamba), which already provides the
"boring" half of a neuron (input-dependent temporal integration).

All blocks map (B, T, C) -> (B, T, C) so they drop into the same
TemporalBlockClassifier used by the temporal-Boolean experiments. A 2D
(B, C) input is treated as a length-1 sequence for easy smoke-testing.

Pure PyTorch, no mamba-ssm/CUDA-kernel dependency (sequential scan over time;
fine at the toy scale we run here). NOTHING here is Mamba-fast; it is Mamba-
faithful enough for an inductive-bias comparison.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_depthwise(x: torch.Tensor, conv: nn.Conv1d, k: int) -> torch.Tensor:
    """x: (B, L, C) -> causal depthwise conv -> (B, L, C)."""
    h = x.transpose(1, 2)
    h = F.pad(h, (k - 1, 0))
    return conv(h).transpose(1, 2)


class SelectiveSSM(nn.Module):
    """Minimal selective (input-dependent) diagonal SSM = the Mamba S6 core.

    Per channel a diagonal linear recurrence whose step size dt and input/output
    projections B, C are functions of the input (selectivity). Sequential scan;
    dt-dependent terms are computed per step to avoid materializing the full
    (B, L, d_inner, d_state) tensor.
    """

    def __init__(self, d_inner: int, d_state: int = 8, dt_rank: int | None = None,
                 chunk: int = 8, dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_inner // 16)
        self.chunk = chunk
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        # A is negative real (stable); stored as log for positivity of -A.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Mamba-style timescale init: bias dt so softplus(bias) is log-uniform in
        # [dt_min, dt_max]. Small-dt channels (~dt_min) decay as exp(-dt*|A|) per
        # step -> memory horizons of hundreds of steps, which is what lets the SSM
        # bridge long gaps. Without this, dt~0.69 and the state forgets in ~1 step.
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))   # inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        proj = self.x_proj(x)                                   # (B, L, dt_rank+2*d_state)
        dt, Bp, Cp = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                       # (B, L, d_inner)
        A = -torch.exp(self.A_log)                             # (d_inner, d_state)

        # Discretized recurrence h_t = a_t * h_{t-1} + b_t, diagonal per (d, state).
        a = torch.exp(dt.unsqueeze(-1) * A)                    # (B, L, d, n) in (0, 1]
        b = (dt * x).unsqueeze(-1) * Bp.unsqueeze(2)           # (B, L, d, n)

        # Chunked associative (Hillis-Steele) scan. Division-free, so it is
        # numerically stable even for fast-decaying states (large |A|); chunking
        # bounds peak memory to (B, chunk, d, n). Within a chunk the linear
        # recurrence is scanned in log2(chunk) vectorized doubling steps using
        # the operator (a1,b1)∘(a2,b2) = (a1*a2, a2*b1 + b2); the running carry
        # h_prev is folded in via the chunk's inclusive a-product.
        K = self.chunk
        h = x.new_zeros(B, d, self.d_state)                    # carried state
        outs = []
        for c0 in range(0, L, K):
            La = a[:, c0:c0 + K]                              # (B, k, d, n)
            Lb = b[:, c0:c0 + K]
            k = La.shape[1]
            shift = 1
            while shift < k:
                a_sh = F.pad(La, (0, 0, 0, 0, shift, 0), value=1.0)[:, :k]
                b_sh = F.pad(Lb, (0, 0, 0, 0, shift, 0), value=0.0)[:, :k]
                Lb = Lb + La * b_sh
                La = La * a_sh
                shift *= 2
            h_chunk = Lb + La * h.unsqueeze(1)               # add carried state
            outs.append(h_chunk)
            h = h_chunk[:, -1]
        H = torch.cat(outs, dim=1)                            # (B, L, d, n)
        y = (H * Cp.unsqueeze(2)).sum(-1)                     # (B, L, d)
        return y + x * self.D


class GatedSSMBlock(nn.Module):
    """Shared Mamba-style scaffold for the SSM block family.

    ``forward`` projects the input into ``n_streams`` inner streams plus a gate,
    applies a causal depthwise conv + SiLU to each stream, lets the subclass
    combine them in ``mix()``, then gates by ``SiLU(z)`` and projects back to
    ``d_model``. A 2D ``(B, C)`` input is treated as a length-1 sequence so the
    same block works for smoke tests and full sequences.

    Subclasses own only the interesting part (the SSM core) via ``mix()``; the
    in/out projections, per-stream conv front-end, and gating live here once.
    """

    def __init__(self, d_model: int, d_inner: int, n_streams: int, conv_k: int):
        super().__init__()
        self.d_inner = d_inner
        self.n_streams = n_streams
        self.conv_k = conv_k
        self.in_proj = nn.Linear(d_model, (n_streams + 1) * d_inner, bias=False)
        self.convs = nn.ModuleList(
            nn.Conv1d(d_inner, d_inner, conv_k, groups=d_inner, bias=True)
            for _ in range(n_streams)
        )
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def mix(self, streams: list[torch.Tensor]) -> torch.Tensor:
        """Combine the conv'd streams into a (B, L, d_inner) pre-gate tensor."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)
        *raw, z = self.in_proj(x).chunk(self.n_streams + 1, dim=-1)
        streams = [F.silu(_causal_depthwise(s, conv, self.conv_k))
                   for s, conv in zip(raw, self.convs)]
        out = self.out_proj(self.mix(streams) * F.silu(z))
        return out.squeeze(1) if squeeze else out


class MambaBlock(GatedSSMBlock):
    """Simplified Mamba block: a single gated selective SSM with a causal conv."""

    def __init__(self, d_model: int, d_inner: int | None = None, d_state: int = 8,
                 dt_rank: int | None = None, conv_k: int = 4, expand: int = 2,
                 chunk: int = 8):
        super().__init__(d_model, d_inner or expand * d_model, n_streams=1, conv_k=conv_k)
        self.ssm = SelectiveSSM(self.d_inner, d_state, dt_rank, chunk=chunk)

    def mix(self, streams: list[torch.Tensor]) -> torch.Tensor:
        return self.ssm(streams[0])


class Plateau(nn.Module):
    """Dendritic Ca2+/NMDA-plateau-like nonlinearity: thresholded + saturating.

    Below a learnable threshold the response is ~0; above it, it rises steeply
    (regenerative) then saturates (plateau). This is the nonlinear, supralinear
    branch response that a *linear* SSM recurrence lacks.
    """

    def __init__(self, d: int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(d))
        self.gain = nn.Parameter(torch.ones(d) * 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(F.softplus(self.gain * (x - self.theta)))


class CoincidenceSSM(GatedSSMBlock):
    """High-rank multiplicative coincidence over two SELECTIVE-MEMORY streams.

    Two independent selective SSMs filter two projections of the input; their
    full-width elementwise product is the NMDA-like coincidence (hundreds of
    multiplicative features, unlike the dendritic block's low-rank G-way soma),
    optionally passed through a regenerative plateau. Unlike the gated-conv
    baseline, the coincident streams carry unbounded *selective* memory, so this
    can solve coincidence tasks whose evidence is separated by gaps longer than
    any fixed conv kernel.

    Ablations: ``combine="sum"`` keeps two memory streams but drops the
    multiplication; ``use_plateau=False`` drops the regenerative nonlinearity.
    """

    def __init__(self, d_model: int, d_inner: int | None = None, d_state: int = 8,
                 dt_rank: int | None = None, conv_k: int = 4, expand: int = 2,
                 use_plateau: bool = True, combine: str = "mult", chunk: int = 8):
        super().__init__(d_model, d_inner or expand * d_model, n_streams=2, conv_k=conv_k)
        self.combine = combine
        self.ssm_a = SelectiveSSM(self.d_inner, d_state, dt_rank, chunk=chunk)
        self.ssm_b = SelectiveSSM(self.d_inner, d_state, dt_rank, chunk=chunk)
        self.plateau = Plateau(self.d_inner) if use_plateau else None

    def mix(self, streams: list[torch.Tensor]) -> torch.Tensor:
        ya, yb = self.ssm_a(streams[0]), self.ssm_b(streams[1])
        y = ya * yb if self.combine == "mult" else ya + yb
        return self.plateau(y) if self.plateau is not None else y


class DendriticSSMBlock(GatedSSMBlock):
    """A tree of nonlinear SSM subunits with a multiplicative soma.

    The inner width is split into ``n_branches`` branches; each runs its own
    selective SSM (independent state, shared params via batch folding) followed
    by a PLATEAU nonlinearity. The soma reads out all branches and, in "mult"
    mode, adds a low-order multiplicative cross-branch binding term.

    Kept as a negative control: in our experiments this tree structure does NOT
    beat ``CoincidenceSSM``'s full-width product, and adding branches did not
    help - i.e. the win comes from the coincidence + plateau, not the topology.

    Ablation flags: ``use_tree=False`` (single branch), ``use_plateau=False``,
    ``soma_mode="add"`` (drop the multiplicative binding).
    """

    def __init__(self, d_model: int, d_inner: int | None = None, n_branches: int = 8,
                 d_state: int = 8, dt_rank: int | None = None, conv_k: int = 4,
                 expand: int = 2, use_tree: bool = True, use_plateau: bool = True,
                 soma_mode: str = "mult", chunk: int = 8):
        G = n_branches if use_tree else 1
        db = max(1, math.ceil((d_inner or expand * d_model) / G))   # per-branch width
        super().__init__(d_model, db * G, n_streams=1, conv_k=conv_k)
        self.G, self.db, self.soma_mode = G, db, soma_mode
        # Shared SSM params across branches; branches differ by state + input
        # slice (folding G into the batch gives independent integration).
        self.branch_ssm = SelectiveSSM(db, d_state, dt_rank, chunk=chunk)
        self.plateau = Plateau(db) if use_plateau else None
        self.soma = nn.Linear(self.d_inner, self.d_inner, bias=True)
        if soma_mode == "mult":
            self.bind_in = nn.Linear(G, G, bias=True)
            self.bind_out = nn.Linear(G, self.d_inner, bias=True)

    def mix(self, streams: list[torch.Tensor]) -> torch.Tensor:
        x_in = streams[0]
        B, L, _ = x_in.shape
        xb = x_in.view(B, L, self.G, self.db).permute(0, 2, 1, 3).reshape(B * self.G, L, self.db)
        a = self.branch_ssm(xb)
        if self.plateau is not None:
            a = self.plateau(a)
        a = a.reshape(B, self.G, L, self.db).permute(0, 2, 1, 3)         # (B, L, G, db)
        out = self.soma(a.reshape(B, L, self.d_inner))
        if self.soma_mode == "mult":
            bsum = a.sum(dim=-1)                                          # (B, L, G)
            out = out + self.bind_out(bsum * self.bind_in(bsum))         # quadratic binding
        return out
