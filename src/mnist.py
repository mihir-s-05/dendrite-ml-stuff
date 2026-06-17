"""MNIST loader from the keras npz (no torchvision dependency)."""

from __future__ import annotations

import os

import numpy as np
import torch

_DEFAULT = os.path.join(os.path.dirname(__file__), os.pardir, "data", "mnist.npz")


def load_mnist(path: str = _DEFAULT):
    """Return flattened, [0,1]-normalized tensors: (Xtr, ytr, Xte, yte)."""
    with np.load(path) as d:
        xtr, ytr = d["x_train"], d["y_train"]
        xte, yte = d["x_test"], d["y_test"]
    xtr = xtr.reshape(len(xtr), -1).astype(np.float32) / 255.0
    xte = xte.reshape(len(xte), -1).astype(np.float32) / 255.0
    return (
        torch.from_numpy(xtr),
        torch.from_numpy(ytr.astype(np.int64)),
        torch.from_numpy(xte),
        torch.from_numpy(yte.astype(np.int64)),
    )
