from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


TASK_T_MAX = {
    "math": 512,
    "logic": 512,
    "code": 768,
    "bias_choice": 384,
    "truthfulness": 384,
    "sycophancy": 384,
    "chinese_exam": 512,
    "safety_policy": 384,
    "unknown": 512,
}


@dataclass
class CotSample:
    id: str
    source_dataset: str
    task_type: str
    question: str
    choices: dict[str, str] | None
    gold_answer: str
    gold_rationale: list[str] = field(default_factory=list)
    essential_propositions: list[str] = field(default_factory=list)
    hidden_cue: str | None = None
    cue_target: str | None = None
    has_hidden_cue: bool = False
    split: str | None = None
    t_max: int | None = None
    needs_human_review: list[str] = field(default_factory=list)
    human_review_notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.t_max is None:
            self.t_max = TASK_T_MAX.get(self.task_type, TASK_T_MAX["unknown"])
        self.question = self.question.strip()
        self.gold_answer = str(self.gold_answer).strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "CotSample":
        return cls(**item)


@dataclass
class ModelOutput:
    sample_id: str
    model: str
    group: str
    prompt_version: str
    raw_output: str
    reasoning: str
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
class PerSampleMetric:
    sample_id: str
    group: str
    model: str
    prompt_version: str
    accuracy: float
    token_cost: float
    density: float
    essential_recall: float
    fluency: float
    entailment_proxy: float
    contradiction: float
    hidden_cue_used: float
    hidden_cue_disclosed: float
    hidden_cue_gap: float
    faithfulness: float
    secrecy: float
    satisfaction: float
    needs_human_review: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
