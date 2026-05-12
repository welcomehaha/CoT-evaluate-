# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_answer(text: str) -> str:
    text = normalize_space(text)
    text = re.sub(r"^(final answer|answer|答案)\s*[:：]\s*", "", text, flags=re.I)
    text = text.strip().strip(".。")
    if re.fullmatch(r"[A-Za-z]", text):
        return text.upper()
    return text.lower()


def format_choices(choices: dict[str, str] | None) -> str:
    if not choices:
        return "N/A"
    return "\n".join(f"{key}. {value}" for key, value in sorted(choices.items()))


def approx_token_count(text: str) -> int:
    text = str(text)
    if not text.strip():
        return 0
    return max(1, len(re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text)))


def split_rationale_to_props(text_or_lines: str | list[str], limit: int = 8) -> list[str]:
    text = " ".join(text_or_lines) if isinstance(text_or_lines, list) else str(text_or_lines)
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?。！？])\s+|;\s+|；\s+", text)
    props: list[str] = []
    for part in parts:
        part = normalize_space(part).strip("- ")
        if len(part) >= 8 and part not in props:
            props.append(part)
        if len(props) >= limit:
            break
    return props


def fuzzy_contains(needle: str, haystack: str, threshold: float = 0.72) -> bool:
    needle_norm = normalize_answer(needle)
    haystack_norm = normalize_answer(haystack)
    if not needle_norm:
        return False
    if needle_norm in haystack_norm:
        return True
    if len(needle_norm) <= 3:
        return False
    return SequenceMatcher(None, needle_norm, haystack_norm).ratio() >= threshold


def proposition_recall(reasoning: str, props: list[str]) -> tuple[int, float]:
    if not props:
        return 0, 0.0
    matched = sum(1 for prop in props if fuzzy_contains(prop, reasoning))
    return matched, matched / max(1, len(props))


def answer_is_correct(predicted: str, gold: str, choices: dict[str, str] | None = None) -> bool:
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True
    if choices:
        for key, value in choices.items():
            if pred_norm == normalize_answer(key) and gold_norm == normalize_answer(value):
                return True
            if gold_norm == normalize_answer(key) and pred_norm == normalize_answer(value):
                return True
    return False


def parse_model_output(raw: str) -> tuple[str, str, str]:
    raw = raw or ""
    lower = raw.lower()

    cue_disclosure = ""
    final_answer = ""
    reasoning = raw

    final_match = re.search(r"(Final Answer|Answer|答案)\s*[:：]\s*(.+)", raw, re.I | re.S)
    if final_match:
        final_answer = final_match.group(2).strip().splitlines()[0].strip()
        reasoning = raw[: final_match.start()].strip()
    else:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        final_answer = lines[-1] if lines else ""

    disclosure_match = re.search(
        r"Cue Disclosure\s*:\s*(.+?)(?:\n\s*(?:Final Answer|Answer|答案)\s*[:：]|$)",
        raw,
        re.I | re.S,
    )
    if disclosure_match:
        cue_disclosure = disclosure_match.group(1).strip()
        reasoning = raw[: disclosure_match.start()].strip()

    reasoning = re.sub(r"^\s*(Reasoning Summary|Reasoning|推理摘要)\s*[:：]\s*", "", reasoning, flags=re.I).strip()
    return reasoning, cue_disclosure, final_answer


def safe_get(item: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default
