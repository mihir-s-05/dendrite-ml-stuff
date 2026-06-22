"""Boolean / feature-binding tasks, mirroring the paper's abstract probes.

The paper encodes binary features into noisy spike trains and asks the neuron
to realize Boolean functions (parity, random balanced functions). We use the
ML analogue: each d-bit pattern is embedded as a +/-1 vector with additive
Gaussian noise, sampled many times, and split into train/test so we measure
genuine generalization (not just truth-table memorization).
"""

from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import torch


def parity_fn(d: int) -> Callable[[np.ndarray], np.ndarray]:
    def f(bits: np.ndarray) -> np.ndarray:
        return bits.sum(axis=-1) % 2
    return f


def subset_parity_fn(d: int, k: int) -> Callable[[np.ndarray], np.ndarray]:
    """Parity over the first k of d bits; the remaining d-k bits are distractors.

    A clean test of *selective* coincidence: the model must bind k specific bits
    across the long-range gap while ignoring the irrelevant ones. Still balanced
    (parity of k bits is balanced regardless of the distractors), so chance is
    50% and a model that integrates everything (no selectivity) cannot cheat.
    """
    if not 0 < k <= d:
        raise ValueError(f"need 0 < k <= d, got k={k}, d={d}")

    def f(bits: np.ndarray) -> np.ndarray:
        return bits[..., :k].sum(axis=-1) % 2
    return f


def random_balanced_boolean(d: int, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
    """A uniformly random balanced Boolean function over d bits."""
    n = 2 ** d
    labels = np.zeros(n, dtype=np.int64)
    labels[: n // 2] = 1
    rng.shuffle(labels)
    powers = (2 ** np.arange(d)[::-1]).astype(np.int64)

    def f(bits: np.ndarray) -> np.ndarray:
        idx = (bits.astype(np.int64) * powers).sum(axis=-1)
        return labels[idx]
    return f


def make_streaming_parity(n_samples: int, seq_len: int, seed: int = 0, p: float = 0.5):
    """Streaming parity (running XOR): an autoregressive state-tracking task.

    Each sample is a random bit stream; the target at every timestep is the
    parity of all bits seen so far. Returns numpy arrays:
        X: (n_samples, seq_len) int64 bits in {0,1}
        y: (n_samples, seq_len) float32 running parity in {0,1}

    A purely linear recurrence (vanilla SSM/Mamba) cannot *maintain* the parity
    bit across time -- it can track a running count, but recovering count mod 2
    is a per-step nonlinearity whose period it must memorize, so it fails to
    length-generalize. A model with genuine inter-step nonlinearity can hold a
    bounded parity state and generalize to longer sequences. This is the
    autoregressive analogue of the long-range coincidence tasks.
    """
    rng = np.random.default_rng(seed)
    bits = (rng.random((n_samples, seq_len)) < p).astype(np.int64)
    parity = (np.cumsum(bits, axis=1) % 2).astype(np.float32)
    return bits, parity


def make_dataset(
    d: int,
    fn: Callable[[np.ndarray], np.ndarray],
    n_per_pattern: int = 64,
    noise_std: float = 0.5,
    test_frac: float = 0.3,
    seed: int = 0,
):
    """Return (X_train, y_train, X_test, y_test) tensors.

    Every one of the 2^d patterns is instantiated n_per_pattern times with
    independent +/-1 -> noise jitter, then split so test patterns are unseen
    noisy realizations of the same underlying function.
    """
    rng = np.random.default_rng(seed)
    patterns = np.array(list(itertools.product([0, 1], repeat=d)), dtype=np.float32)
    labels = fn(patterns)

    X = np.repeat(patterns, n_per_pattern, axis=0)
    y = np.repeat(labels, n_per_pattern, axis=0)
    signed = (X * 2.0 - 1.0) + rng.normal(0.0, noise_std, size=X.shape).astype(np.float32)

    perm = rng.permutation(len(signed))
    signed, y = signed[perm], y[perm]
    n_test = int(len(signed) * test_frac)

    Xte, yte = signed[:n_test], y[:n_test]
    Xtr, ytr = signed[n_test:], y[n_test:]
    to = lambda a, dt: torch.tensor(a, dtype=dt)
    return (
        to(Xtr, torch.float32),
        to(ytr, torch.float32),
        to(Xte, torch.float32),
        to(yte, torch.float32),
    )


def make_temporal_boolean_dataset(
    d: int,
    fn: Callable[[np.ndarray], np.ndarray],
    n_per_pattern: int = 32,
    n_time: int = 48,
    axons_per_bit: int = 8,
    jitter_std: float = 1.5,
    spike_width: float = 0.75,
    background_rate: float = 0.002,
    test_frac: float = 0.3,
    seed: int = 0,
    active_frac: tuple[float, float] | None = None,
):
    """Return noisy spike/event sequences for Boolean feature-binding tasks.

    Each bit owns ON and OFF afferent populations. For a given pattern, the
    active population emits one jittered spike per axon at axon-specific target
    times that tile the full sequence. This is still lightweight PyTorch data,
    but it preserves the paper's key setup better than static +/-1 vectors:
    redundant afferents, timed spikes, jitter, and a later decision window.

    Shape:
        X: `(samples, n_time, d * 2 * axons_per_bit)`
        y: `(samples,)`
    """
    if n_time < 4:
        raise ValueError("n_time must be at least 4")
    if axons_per_bit < 1:
        raise ValueError("axons_per_bit must be at least 1")

    rng = np.random.default_rng(seed)
    patterns = np.array(list(itertools.product([0, 1], repeat=d)), dtype=np.int64)
    labels = fn(patterns).astype(np.float32)
    if active_frac is None:
        # Default: spikes tile the full sequence (late axons land in the readout).
        lo, hi = 2.0, float(n_time - 3)
    else:
        # Long-range mode: confine all evidence to an early window, leaving a gap
        # before the decision window so fixed-kernel convs cannot bridge it.
        lo = max(1.0, active_frac[0] * n_time)
        hi = max(lo + 1.0, active_frac[1] * n_time)
    base_times = np.linspace(lo, hi, axons_per_bit, dtype=np.float32)
    n_channels = d * 2 * axons_per_bit

    # Expand patterns to samples (vectorized; no Python per-sample loops).
    bits = np.repeat(patterns, n_per_pattern, axis=0)          # (S, d) in {0,1}
    y = np.repeat(labels, n_per_pattern, axis=0)               # (S,)
    S = bits.shape[0]

    bit_idx = np.arange(d)[None, :, None]                      # (1, d, 1)
    axon_idx = np.arange(axons_per_bit)[None, None, :]         # (1, 1, A)
    polarity = bits[:, :, None]                                # (S, d, 1) ON/OFF
    # Channel layout: ((bit * 2 + polarity) * axons_per_bit + axon).
    channel = (bit_idx * 2 + polarity) * axons_per_bit + axon_idx          # (S, d, A)
    channel = np.broadcast_to(channel, (S, d, axons_per_bit))

    target = base_times[None, None, :] + rng.normal(0.0, jitter_std, size=(S, d, axons_per_bit))
    t = np.clip(np.rint(target).astype(np.int64), 0, n_time - 1)           # (S, d, A)

    X = np.zeros((S, n_time, n_channels), dtype=np.float32)
    if background_rate > 0:
        X[:] = (rng.random((S, n_time, n_channels)) < background_rate).astype(np.float32)

    s_idx = np.broadcast_to(np.arange(S)[:, None, None], (S, d, axons_per_bit))
    sf, cf, tf = s_idx.ravel(), channel.ravel(), t.ravel()
    # Neighbor "spike width" first (np.maximum.at handles duplicate targets),
    # then stamp the spike centers to 1.0 so a center always dominates.
    if spike_width > 0:
        for off in (-1, 1):
            tn = tf + off
            ok = (tn >= 0) & (tn < n_time)
            np.maximum.at(X, (sf[ok], tn[ok], cf[ok]), np.float32(spike_width))
    X[sf, tf, cf] = 1.0

    perm = rng.permutation(S)
    X, y = X[perm], y[perm]
    n_test = int(S * test_frac)
    Xte, yte = X[:n_test], y[:n_test]
    Xtr, ytr = X[n_test:], y[n_test:]
    return (
        torch.from_numpy(Xtr),
        torch.from_numpy(ytr),
        torch.from_numpy(Xte),
        torch.from_numpy(yte),
    )
