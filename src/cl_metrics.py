"""Standard continual-learning metrics from an R-matrix (Lopez-Paz & Ranzato,
GEM 2017). Computed on a bounded per-domain score (next-token accuracy), so
they don't blow up the way perplexity does on a forgotten domain.

R[i, j] = score on domain j after training through domain i.
base[j] = score on domain j at initialization (before any training).
"""

from __future__ import annotations

import numpy as np


def cl_metrics(R: np.ndarray, base: np.ndarray) -> dict:
    T = R.shape[0]
    final = R[T - 1]                       # performance after the whole sequence
    avg = float(final.mean())              # average final accuracy (higher better)
    # Backward transfer: how much earlier tasks changed by the end (neg = forgot).
    bwt = float(np.mean([R[T - 1, j] - R[j, j] for j in range(T - 1)])) if T > 1 else 0.0
    # Forward transfer: zero-shot gain on a task before training it, vs init.
    fwt = float(np.mean([R[j - 1, j] - base[j] for j in range(1, T)])) if T > 1 else 0.0
    # Forgetting = positive number for "got worse" (i.e. -BWT, clipped at 0 per task).
    forgetting = float(np.mean([max(0.0, R[j, j] - R[T - 1, j]) for j in range(T - 1)])) if T > 1 else 0.0
    return {"avg_acc": avg, "bwt": bwt, "fwt": fwt, "forgetting": forgetting}
