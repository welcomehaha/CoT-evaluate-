# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import MitigationSample


@dataclass
class DynamicPenaltyConfig:
    eta_difficulty: float = 0.25
    eta_risk: float = 0.30
    eta_cue_presence: float = 0.30
    eta_uncertainty: float = 0.15
    min_multiplier: float = 0.15
    max_multiplier: float = 1.20

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DynamicPenaltyConfig":
        if not data:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: float(value) for key, value in data.items() if key in allowed})


def risk_quality(sample: MitigationSample, cfg: DynamicPenaltyConfig) -> float:
    cue_presence = 1.0 if sample.has_hidden_cue else 0.0
    q = (
        cfg.eta_difficulty * clamp(sample.difficulty)
        + cfg.eta_risk * clamp(sample.risk)
        + cfg.eta_cue_presence * cue_presence
        + cfg.eta_uncertainty * clamp(sample.uncertainty)
    )
    return clamp(q)


def dynamic_lambda(base_lambda_l: float, sample: MitigationSample, cfg: DynamicPenaltyConfig) -> float:
    q = risk_quality(sample, cfg)
    multiplier = 1.0 - q
    multiplier = max(cfg.min_multiplier, min(cfg.max_multiplier, multiplier))
    return float(base_lambda_l * multiplier)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
