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


class PlateauRecurrentSSM(nn.Module):
    """Selective SSM with the regenerative nonlinearity INSIDE the recurrence.

    The linear ``SelectiveSSM`` integrates ``h_t = a_t*h_{t-1} + b_t`` and only
    squashes the *output*. That cannot length-generalize a state machine like
    parity: the readout learns a length-specific count->parity map. Here the
    nonlinearity is applied to the carried state every step, and the per-step
    multiplicative factor is a SIGNED, input-dependent gate, so the state can
    *flip* (the multiplicative dendritic selectivity, now in the loop):

        m_t = a_t * g_t,    g_t = tanh(W_g x_t) in (-1, 1)
        h_t = tanh(m_t * h_{t-1} + b_t)

    ``a_t = exp(dt*A) in (0,1]`` is the usual selective decay (bounded -> stable);
    ``g_t`` lets the recurrence sign-invert (parity = repeated sign flip); the
    ``tanh`` state map is the bounded regenerative plateau that keeps the state
    from growing with length. The gate bias inits near +1 so the block behaves
    like a plain selective SSM at init and learns to flip from there.

    Sequential scan (the nonlinearity breaks associativity, so no chunked
    parallel scan); per-step terms are built inside the loop to keep peak memory
    at (B, d_inner, d_state) rather than (B, L, d_inner, d_state).
    """

    def __init__(self, d_inner: int, d_state: int = 8, dt_rank: int | None = None,
                 dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_inner // 16)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        self.g_proj = nn.Linear(d_inner, d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            # Bias the gate near +1 (tanh(2)~0.96) so the recurrence starts as a
            # plain selective decay and learns to flip sign during training.
            self.g_proj.weight.mul_(0.1)
            self.g_proj.bias.fill_(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        proj = self.x_proj(x)
        dt, Bp, Cp = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                      # (B, L, d_inner)
        g = torch.tanh(self.g_proj(x))                         # (B, L, d_inner) in (-1,1)
        A = -torch.exp(self.A_log)                             # (d_inner, d_state)

        h = x.new_zeros(B, d, self.d_state)
        outs = []
        for t in range(L):
            a_t = torch.exp(dt[:, t].unsqueeze(-1) * A)        # (B, d, n) in (0,1]
            m_t = a_t * g[:, t].unsqueeze(-1)                  # signed factor in (-1,1)
            b_t = (dt[:, t] * x[:, t]).unsqueeze(-1) * Bp[:, t].unsqueeze(1)
            h = torch.tanh(m_t * h + b_t)                      # nonlinearity in the loop
            outs.append((h * Cp[:, t].unsqueeze(1)).sum(-1))   # (B, d)
        y = torch.stack(outs, dim=1)                           # (B, L, d)
        return y + x * self.D


class RotationRecurrentSSM(nn.Module):
    """Selective COMPLEX recurrence: an input-dependent ROTATION in the loop.

    ``PlateauRecurrentSSM`` carries a real state whose per-step factor is a
    signed scalar -- a 2-cycle (parity), nothing more, because a diagonal real
    recurrence cannot rotate. Here each (channel, state) carries a 2D / complex
    state ``z = (zr, zi)`` whose per-step eigenvalue is ``rho * e^{i*theta}``:

        z_t = plateau_radial( rho_t * R(theta_t) z_{t-1} + b_t )

    ``theta_t`` is an input-dependent rotation angle, so the recurrence can learn
    to advance phase by ``2*pi/k`` per increment and realize an arbitrary k-cycle
    (mod-k counter); ``rho_t = exp(dt*A) in (0,1]`` is the selective magnitude
    decay. The radial plateau ``tanh(gain*|z|)`` (gain>1) is regenerative near 0
    and saturating for large |z|, giving a stable NONZERO fixed radius -- an
    attracting ring -- so the rotor persists across long gaps and the state stays
    bounded, which is what lets it length-generalize. Phase is preserved (the
    nonlinearity scales magnitude only), so the rotation stays exact.

    The angle inits near 0 (so the block starts as a plain real decaying SSM and
    learns to rotate from there). Sequential scan; per-step terms built in-loop.
    """

    def __init__(self, d_inner: int, d_state: int = 8, dt_rank: int | None = None,
                 dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_inner // 16)
        # proj -> [dt, B(drive), C_re, C_im]
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 3 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        self.theta_proj = nn.Linear(d_inner, d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        # Radial plateau gain per channel; softplus(raw)+1 > 1 => attracting ring.
        self.gain_raw = nn.Parameter(torch.zeros(d_inner))

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            # Start near zero rotation: behaves like a plain real decaying SSM,
            # then learns the per-increment angle (e.g. 2*pi/k) during training.
            self.theta_proj.weight.mul_(0.1)
            self.theta_proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        proj = self.x_proj(x)
        dt, Bp, Cr, Ci = torch.split(
            proj, [self.dt_rank, self.d_state, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                      # (B, L, d_inner)
        theta = self.theta_proj(x)                             # (B, L, d_inner)
        A = -torch.exp(self.A_log)                             # (d_inner, d_state)
        gain = (F.softplus(self.gain_raw) + 1.0).unsqueeze(-1)  # (d_inner, 1) > 1

        zr = x.new_zeros(B, d, self.d_state)
        zi = x.new_zeros(B, d, self.d_state)
        outs = []
        for t in range(L):
            rho = torch.exp(dt[:, t].unsqueeze(-1) * A)        # (B, d, n) in (0,1]
            th = theta[:, t].unsqueeze(-1)                     # (B, d, 1)
            cos, sin = torch.cos(th), torch.sin(th)
            b_t = (dt[:, t] * x[:, t]).unsqueeze(-1) * Bp[:, t].unsqueeze(1)
            zr_lin = rho * (cos * zr - sin * zi) + b_t         # rotate + decay + drive
            zi_lin = rho * (sin * zr + cos * zi)
            mag = torch.sqrt(zr_lin * zr_lin + zi_lin * zi_lin + 1e-8)
            scale = torch.tanh(gain * mag) / mag               # radial plateau (phase-safe)
            zr, zi = zr_lin * scale, zi_lin * scale
            yt = (zr * Cr[:, t].unsqueeze(1) + zi * Ci[:, t].unsqueeze(1)).sum(-1)
            outs.append(yt)                                    # (B, d)
        y = torch.stack(outs, dim=1)                           # (B, L, d)
        return y + x * self.D


class QuantizedRotationSSM(nn.Module):
    """Rotation recurrence whose angle is QUANTIZED to an exact rational grid.

    ``RotationRecurrentSSM`` rotates by a *free* learned angle, so it represents
    mod-k but drifts at long lengths: nothing forces the angle to be exactly
    ``2*pi/k``, and the phase error grows ~ length * delta_theta (final-position
    acc falls *below* chance -- the drift signature). Here the per-step angle is
    snapped, per (channel, step), to the nearest multiple of ``2*pi/n_bins`` by
    straight-through ROUNDING: the forward rotation is therefore an exact grid
    angle, so if ``n_bins`` is a multiple of k the model can realize the exact
    ``2*pi/k`` and accumulate ZERO drift -> length generalizes arbitrarily.
    Gradients flow as if unrounded, so the input-dependent angle stays trainable.
    Rounding (rather than a categorical over n_bins) keeps the angle projection
    at ``d_inner^2`` cost, so grid resolution is free and d_inner is not starved
    -- the categorical variant spent the whole budget on ``d_inner*n_bins``
    logits, shrinking the recurrence and crippling it at fine grids.

    Everything else matches ``RotationRecurrentSSM`` (selective magnitude decay,
    phase-safe radial plateau, complex readout).
    """

    def __init__(self, d_inner: int, d_state: int = 8, dt_rank: int | None = None,
                 n_bins: int = 12, dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_inner // 16)
        self.n_bins = n_bins
        self.delta = 2 * math.pi / n_bins                      # angle grid spacing
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 3 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        # Continuous per-channel angle (d_inner^2 cost, like the free rotor),
        # snapped to the grid by straight-through rounding in forward().
        self.theta_proj = nn.Linear(d_inner, d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.gain_raw = nn.Parameter(torch.zeros(d_inner))

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            # Start near zero rotation (snaps to angle 0): plain real decaying SSM,
            # then learns which input advances the angle to an exact grid multiple.
            self.theta_proj.weight.mul_(0.1)
            self.theta_proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        proj = self.x_proj(x)
        dt, Bp, Cr, Ci = torch.split(
            proj, [self.dt_rank, self.d_state, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                      # (B, L, d_inner)
        A = -torch.exp(self.A_log)                             # (d_inner, d_state)
        gain = (F.softplus(self.gain_raw) + 1.0).unsqueeze(-1)  # (d_inner, 1) > 1

        # Straight-through rounding: snap the continuous angle to the nearest grid
        # multiple of delta in the forward pass (so the realized rotation is an
        # exact rational angle -> zero drift); gradient flows as if unrounded.
        theta = self.theta_proj(x)                            # (B, L, d) continuous
        theta_q = torch.round(theta / self.delta) * self.delta
        theta = theta + (theta_q - theta).detach()            # ST exact-angle snap
        cos_a = torch.cos(theta)                              # (B, L, d)
        sin_a = torch.sin(theta)

        zr = x.new_zeros(B, d, self.d_state)
        zi = x.new_zeros(B, d, self.d_state)
        outs = []
        for t in range(L):
            rho = torch.exp(dt[:, t].unsqueeze(-1) * A)        # (B, d, n) in (0,1]
            cos = cos_a[:, t].unsqueeze(-1)                    # (B, d, 1)
            sin = sin_a[:, t].unsqueeze(-1)
            b_t = (dt[:, t] * x[:, t]).unsqueeze(-1) * Bp[:, t].unsqueeze(1)
            zr_lin = rho * (cos * zr - sin * zi) + b_t         # exact rotate + decay + drive
            zi_lin = rho * (sin * zr + cos * zi)
            mag = torch.sqrt(zr_lin * zr_lin + zi_lin * zi_lin + 1e-8)
            scale = torch.tanh(gain * mag) / mag               # radial plateau (phase-safe)
            zr, zi = zr_lin * scale, zi_lin * scale
            yt = (zr * Cr[:, t].unsqueeze(1) + zi * Ci[:, t].unsqueeze(1)).sum(-1)
            outs.append(yt)                                    # (B, d)
        y = torch.stack(outs, dim=1)                           # (B, L, d)
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


class RecurrentDendriticBlock(GatedSSMBlock):
    """Gated block wrapping ``PlateauRecurrentSSM`` (nonlinearity in the loop).

    Same Mamba-style scaffold as the other blocks so it drops into the harness;
    the only difference from ``MambaBlock`` is the nonlinear, sign-flipping
    recurrence at its core. ``chunk`` is accepted for call-site uniformity but
    ignored (the scan is sequential).
    """

    def __init__(self, d_model: int, d_inner: int | None = None, d_state: int = 8,
                 dt_rank: int | None = None, conv_k: int = 4, expand: int = 2,
                 chunk: int = 8):
        super().__init__(d_model, d_inner or expand * d_model, n_streams=1, conv_k=conv_k)
        self.ssm = PlateauRecurrentSSM(self.d_inner, d_state, dt_rank)

    def mix(self, streams: list[torch.Tensor]) -> torch.Tensor:
        return self.ssm(streams[0])


class RotationRecurrentBlock(GatedSSMBlock):
    """Gated block wrapping ``RotationRecurrentSSM`` (selective rotation in loop).

    Same scaffold as the other blocks; the core is the complex/rotation
    recurrence that generalizes the signed-gate 2-cycle to arbitrary k-cycles.
    ``chunk`` is accepted for call-site uniformity but ignored (sequential scan).
    """

    def __init__(self, d_model: int, d_inner: int | None = None, d_state: int = 8,
                 dt_rank: int | None = None, conv_k: int = 4, expand: int = 2,
                 chunk: int = 8):
        super().__init__(d_model, d_inner or expand * d_model, n_streams=1, conv_k=conv_k)
        self.ssm = RotationRecurrentSSM(self.d_inner, d_state, dt_rank)

    def mix(self, streams: list[torch.Tensor]) -> torch.Tensor:
        return self.ssm(streams[0])


class QuantizedRotationBlock(GatedSSMBlock):
    """Gated block wrapping ``QuantizedRotationSSM`` (exact-grid rotation in loop).

    Same scaffold as the other blocks; the core rotates by an angle snapped to a
    rational grid (``n_bins``), removing the phase drift that limits the free-
    angle rotation block. ``chunk`` is accepted for uniformity but ignored.
    """

    def __init__(self, d_model: int, d_inner: int | None = None, d_state: int = 8,
                 dt_rank: int | None = None, conv_k: int = 4, expand: int = 2,
                 n_bins: int = 12, chunk: int = 8):
        super().__init__(d_model, d_inner or expand * d_model, n_streams=1, conv_k=conv_k)
        self.ssm = QuantizedRotationSSM(self.d_inner, d_state, dt_rank, n_bins=n_bins)

    def mix(self, streams: list[torch.Tensor]) -> torch.Tensor:
        return self.ssm(streams[0])
