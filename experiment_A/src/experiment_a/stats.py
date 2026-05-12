# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import random
from typing import Iterable

import numpy as np


def bootstrap_ci(values: Iterable[float], n_boot: int = 2000, seed: int = 20260511) -> tuple[float, float, float]:
    vals = np.array(list(values), dtype=float)
    if len(vals) == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [float(vals[rng.randrange(len(vals))]) for _ in range(len(vals))]
        means.append(float(np.mean(sample)))
    return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cohens_d(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.array(list(a), dtype=float)
    y = np.array(list(b), dtype=float)
    if len(x) < 2 or len(y) < 2:
        return 0.0
    pooled = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    if pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)
