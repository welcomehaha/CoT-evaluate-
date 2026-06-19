# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


TASK_TYPES = {"code_repair", "reasoning", "summary", "safety"}
MONITOR_INPUTS = {"output_only", "cot_only", "cot_output", "tool_log"}


@dataclass
class HackingTask:
    id: str
    source_dataset: str
    task_type: str
    task: str
    choices: dict[str, str] | None = None
    gold_answer: str | None = None
    gold_output: str | None = None
    gold_rationale: list[str] = field(default_factory=list)
    essential_propositions: list[str] = field(default_factory=list)
    hidden_tests: list[str] = field(default_factory=list)
    public_tests: list[str] = field(default_factory=list)
    risk_label: str | None = None
    expected_behavior: str = ""
    shortcut_opportunity: str | None = None
    shortcut_type: str | None = None
    forbidden_actions: list[str] = field(default_factory=list)
    monitor_visible_fields: list[str] = field(default_factory=lambda: ["task", "reasoning", "final_output"])
    split: str | None = None
    t_max: int = 768
    needs_human_audit: list[str] = field(default_factory=list)
    human_audit_notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "HackingTask":
        return cls(**item)


@dataclass
class ModelOutput:
    sample_id: str
    model: str
    group: str
    raw_output: str
    reasoning: str
    shortcut_disclosure: str
    final_output: str
    output_tokens: int | None = None
    input_tokens: int | None = None
    latency_s: float | None = None
    tool_log: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "ModelOutput":
        return cls(**item)


@dataclass
class AuditResult:
    sample_id: str
    model: str
    group: str
    task_type: str
    shortcut_type: str | None
    task_score: float
    groundtruth_score: float
    monitor_score: float
    hack_detected: float
    hack_type: str
    behavioral_failure: float
    shortcut_used: float
    shortcut_disclosed: float
    concealment: float
    reward_score_gap: float
    cot_tokens: float
    monitor_input: str = "cot_output"
    monitor_detected: float = 0.0
    monitor_confidence: float = 0.0
    monitor_false_negative: float = 0.0
    monitor_false_positive: float = 0.0
    output_tokens: float = 0.0
    input_tokens: float = 0.0
    latency_s: float = 0.0
    human_audit_completed: bool = False
    needs_human_audit: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
