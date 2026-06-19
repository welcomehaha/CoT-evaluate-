# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MitigationSample:
    id: str
    source_dataset: str
    task_type: str
    task: str
    choices: dict[str, str] | None = None
    gold_answer: str | None = None
    gold_rationale: list[str] = field(default_factory=list)
    essential_propositions: list[str] = field(default_factory=list)
    hidden_cue: str | None = None
    cue_target: str | None = None
    has_hidden_cue: bool = False
    shortcut_type: str | None = None
    difficulty: float = 0.5
    risk: float = 0.0
    uncertainty: float = 0.5
    language: str = "en"
    split: str | None = None
    t_max: int = 768
    needs_human_review: list[str] = field(default_factory=list)
    human_review_notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "MitigationSample":
        return cls(**item)


@dataclass
class ModelOutput:
    sample_id: str
    model: str
    group: str
    raw_output: str
    reasoning: str
    cue_disclosure: str
    final_answer: str
    output_tokens: int | None = None
    input_tokens: int | None = None
    latency_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "ModelOutput":
        return cls(**item)


@dataclass
class MitigationMetric:
    sample_id: str
    model: str
    group: str
    task_type: str
    difficulty_bucket: str
    risk_bucket: str
    has_hidden_cue: bool
    accuracy: float
    F: float
    Se: float
    verbalization_recall: float
    token_cost: float
    S: float
    over_confession: float
    U: float
    key_premise_recall: float
    structured_score: float
    dynamic_lambda_L: float
    output_tokens: float = 0.0
    input_tokens: float = 0.0
    latency_s: float = 0.0
    human_review_completed: bool = False
    needs_human_review: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
