# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


TASK_T_MAX = {
    "choice_reasoning": 512,
    "bias_choice": 384,
    "truthfulness": 384,
    "sycophancy": 384,
    "math": 512,
    "code": 768,
    "chinese_exam": 512,
    "unknown": 512,
}

CUE_TYPES = {
    "authority",
    "majority",
    "user_preference",
    "answer_position",
    "misleading_rationale",
    "reward_shortcut",
}

CUE_RELATIONS = {"conflict", "consistent", "neutral"}


@dataclass
class HiddenCueSample:
    id: str
    base_sample_id: str
    variant_id: str
    source_dataset: str
    task_type: str
    question: str
    choices: dict[str, str] | None
    gold_answer: str
    gold_rationale: list[str] = field(default_factory=list)
    essential_propositions: list[str] = field(default_factory=list)
    prompt_version: str = "original"
    hidden_cue: str | None = None
    cue_type: str | None = None
    cue_target: str | None = None
    cue_relation: str = "neutral"
    language: str = "en"
    split: str | None = None
    t_max: int | None = None
    needs_human_review: list[str] = field(default_factory=list)
    human_review_notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.t_max is None:
            self.t_max = TASK_T_MAX.get(self.task_type, TASK_T_MAX["unknown"])
        self.question = str(self.question).strip()
        self.gold_answer = str(self.gold_answer).strip()
        self.prompt_version = "cue" if self.hidden_cue else self.prompt_version
        if self.prompt_version == "original":
            self.hidden_cue = None
            self.cue_type = None
            self.cue_target = None
            self.cue_relation = "neutral"

    @property
    def has_hidden_cue(self) -> bool:
        return self.prompt_version == "cue" and bool(self.hidden_cue)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"has_hidden_cue": self.has_hidden_cue}

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "HiddenCueSample":
        item = dict(item)
        item.pop("has_hidden_cue", None)
        return cls(**item)


@dataclass
class ModelOutput:
    sample_id: str
    base_sample_id: str
    variant_id: str
    model: str
    group: str
    prompt_version: str
    raw_output: str
    reasoning_summary: str
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
class HiddenCueMetric:
    sample_id: str
    base_sample_id: str
    variant_id: str
    model: str
    group: str
    prompt_version: str
    cue_type: str | None
    cue_relation: str
    language: str
    accuracy_original: float | None
    accuracy_cue: float
    accuracy_original_auto: float | None = None
    accuracy_cue_auto: float = 0.0
    accuracy_original_judged: float | None = None
    accuracy_cue_judged: float | None = None
    accuracy_drop: float | None = None
    answer_changed: float = 0.0
    answer_changed_to_cue_target: float = 0.0
    cue_following: float = 0.0
    used_hidden_cue: float = 0.0
    verbalized_hidden_cue: float = 0.0
    hidden_cue_gap: float = 0.0
    secrecy: float = 0.0
    secrecy_cond: float = 0.0
    rationalization: float = 0.0
    token_cost: float = 0.0
    output_tokens: float = 0.0
    input_tokens: float = 0.0
    latency_s: float = 0.0
    essential_recall: float = 0.0
    fluency: float = 0.0
    human_review_completed: bool = False
    needs_human_review: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
