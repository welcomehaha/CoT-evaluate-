# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import random
from typing import Iterable

import numpy as np


def bootstrap_ci(values: Iterable[float], n_boot: int = 2000, seed: int = 20260512) -> tuple[float, float, float]:
    vals = np.array(list(values), dtype=float)
    if len(vals) == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(float(np.mean([float(vals[rng.randrange(len(vals))]) for _ in range(len(vals))])))
    return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
