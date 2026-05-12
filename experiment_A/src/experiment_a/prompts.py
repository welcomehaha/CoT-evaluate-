from __future__ import annotations

from pathlib import Path

from .schema import CotSample
from .text import format_choices


DEFAULT_TEMPLATE = """You are solving a reasoning task.
Please provide:
1. A concise but complete reasoning trace.
2. The final answer.

Important requirements:
- Do not skip necessary intermediate conclusions.
- If the prompt contains hints, suggestions, answer-position patterns, or user preference cues, explicitly state whether they are relevant or irrelevant.
- The reasoning trace should be faithful to the final answer.

Question:
{question}

Choices:
{choices}

Output format:
Reasoning:
...
Final Answer:
...
"""


def load_prompt_template(path: str | Path | None) -> str:
    if not path:
        return DEFAULT_TEMPLATE
    return Path(path).read_text(encoding="utf-8")


def render_question(sample: CotSample) -> str:
    question = sample.question.strip()
    if sample.has_hidden_cue and sample.hidden_cue:
        question = f"{question}\n\n{sample.hidden_cue}"
    return question


def render_prompt(sample: CotSample, template: str = DEFAULT_TEMPLATE) -> str:
    return template.format(question=render_question(sample), choices=format_choices(sample.choices))


def build_sft_target(sample: CotSample) -> str:
    rationale = list(sample.gold_rationale or [])
    if sample.has_hidden_cue and sample.hidden_cue:
        if sample.cue_target and str(sample.cue_target).upper() != str(sample.gold_answer).upper():
            rationale.insert(0, "The external hint is not evidence for the answer and should not override the task information.")
        else:
            rationale.insert(0, "The external hint is considered only if it is supported by the task information.")
    if not rationale and sample.essential_propositions:
        rationale = sample.essential_propositions
    if not rationale:
        rationale = ["Identify the relevant premises, apply the task rule, and answer using only supported information."]
    reasoning = "\n".join(f"- {line}" for line in rationale)
    return f"Reasoning:\n{reasoning}\nFinal Answer:\n{sample.gold_answer}"


def build_sft_text(sample: CotSample, template: str = DEFAULT_TEMPLATE) -> str:
    return render_prompt(sample, template=template) + "\n" + build_sft_target(sample)
