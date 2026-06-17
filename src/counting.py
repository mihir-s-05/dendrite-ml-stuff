"""Fair-comparison utilities: count params and size blocks to a budget."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def size_to_budget(
    make_block: Callable[[int], nn.Module],
    target_params: int,
    lo: int = 2,
    hi: int = 8192,
) -> int:
    """Binary-search the smallest hidden size whose block params >= target.

    `make_block(hidden)` must build the FFN block for a given hidden size.
    Returns the hidden size whose param count is closest to target.
    """
    best_h, best_err = lo, float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        p = count_params(make_block(mid))
        err = abs(p - target_params)
        if err < best_err:
            best_err, best_h = err, mid
        if p < target_params:
            lo = mid + 1
        else:
            hi = mid - 1
    return best_h
