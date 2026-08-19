"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch
from kale.utils.seed import set_seed as _kale_set_seed


def set_seed(seed: int = 2026) -> None:
    """Seed Python, NumPy and PyTorch, and make CUDA kernels deterministic.

    Delegates to :func:`kale.utils.seed.set_seed`. Exact reproducibility still
    depends on matching software and hardware.
    """
    _kale_set_seed(seed)


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker.

    Pass as ``worker_init_fn`` so patch subsampling is reproducible when loading
    with ``num_workers > 0``, where each worker otherwise inherits an unseeded RNG.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
