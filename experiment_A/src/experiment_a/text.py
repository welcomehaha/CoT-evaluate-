# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any


FINAL_PATTERNS = [
    re.compile(r"(?:Final Answer|Answer|\u7b54\u6848)\s*[:\uff1a]\s*(.+)", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:The answer is)\s+(.+)", re.IGNORECASE | re.DOTALL),
]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_answer(text: str) -> str:
    text = normalize_space(text).strip()
    text = re.sub(r"^(?:final answer|answer|\u7b54\u6848)\s*[:\uff1a]\s*", "", text, flags=re.I)
    text = text.strip().strip(".;, \n\t\u3002\uff0c\uff1b\uff1a")
    if re.fullmatch(r"[A-Za-z]", text):
        return text.upper()
    return text.lower()


def parse_output(raw: str) -> tuple[str, str]:
    raw = raw or ""
    answer = ""
    for pattern in FINAL_PATTERNS:
        match = pattern.search(raw)
        if match:
            answer = match.group(1).strip()
            break
    if not answer:
        non_empty = [line.strip() for line in raw.splitlines() if line.strip()]
        answer = non_empty[-1] if non_empty else ""
    answer = answer.splitlines()[0].strip()

    reasoning = raw
    lower = raw.lower()
    markers = ["final answer", "answer:", "\u7b54\u6848"]
    marker_positions = [lower.find(marker) for marker in markers if lower.find(marker) >= 0]
    if marker_positions:
        reasoning = raw[: min(marker_positions)]
    reasoning = re.sub(
        r"^\s*(?:Reasoning|Reasoning Summary|\u63a8\u7406|\u63a8\u7406\u6458\u8981)\s*[:\uff1a]\s*",
        "",
        reasoning,
        flags=re.I,
    )
    return reasoning.strip(), answer.strip()


def format_choices(choices: dict[str, str] | None) -> str:
    if not choices:
        return "N/A"
    return "\n".join(f"{key}. {value}" for key, value in sorted(choices.items()))


def approx_token_count(text: str) -> int:
    text = str(text)
    if not text.strip():
        return 0
    tokens = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text)
    return max(1, len(tokens))


def split_rationale_to_props(text_or_lines: str | list[str], limit: int = 8) -> list[str]:
    if isinstance(text_or_lines, list):
        text = " ".join(str(x) for x in text_or_lines)
    else:
        text = str(text_or_lines)
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+|;\s+|\uff1b\s*", text)
    props = []
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
    ratio = SequenceMatcher(None, needle_norm, haystack_norm).ratio()
    return ratio >= threshold


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
            key_norm = normalize_answer(key)
            value_norm = normalize_answer(value)
            if key_norm == gold_norm and pred_norm == value_norm:
                return True
            if pred_norm == key_norm and gold_norm == value_norm:
                return True
    return False


def load_yaml(path: str):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_get(item: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default
