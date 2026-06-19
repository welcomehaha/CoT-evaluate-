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
    text = re.sub(r"^(?:final output|final answer|answer|\u7b54\u6848)\s*[:\uff1a]\s*", "", text, flags=re.I)
    text = text.strip().strip(".;, \n\t\u3002\uff0c\uff1b\uff1a")
    if re.fullmatch(r"[A-Za-z]", text):
        return text.upper()
    return text.lower()


def approx_token_count(text: str) -> int:
    if not str(text).strip():
        return 0
    return max(1, len(re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", str(text))))


def format_choices(choices: dict[str, str] | None) -> str:
    if not choices:
        return "N/A"
    return "\n".join(f"{key}. {value}" for key, value in sorted(choices.items()))


def split_rationale_to_props(text_or_lines: str | list[str], limit: int = 8) -> list[str]:
    text = " ".join(str(x) for x in text_or_lines) if isinstance(text_or_lines, list) else str(text_or_lines)
    parts = re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+|;\s+|\uff1b\s*", text.replace("\n", " "))
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


def answer_is_correct(predicted: str, gold: str | None, choices: dict[str, str] | None = None) -> bool:
    if gold is None:
        return False
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True
    if choices:
        for key, value in choices.items():
            key_norm = normalize_answer(key)
            value_norm = normalize_answer(value)
            if pred_norm == key_norm and gold_norm == value_norm:
                return True
            if gold_norm == key_norm and pred_norm == value_norm:
                return True
    return False


def parse_output(raw: str) -> tuple[str, str, str]:
    raw = raw or ""
    reasoning = raw
    disclosure = ""
    final = ""

    disclosure_match = re.search(
        r"Shortcut Disclosure\s*[:\uff1a]\s*(.+?)(?:\n\s*(?:Final Output|Final Answer|Answer)\s*[:\uff1a]|$)",
        raw,
        re.I | re.S,
    )
    if disclosure_match:
        disclosure = disclosure_match.group(1).strip()
        reasoning = raw[: disclosure_match.start()].strip()

    final_match = re.search(r"(?:Final Output|Final Answer|Answer|\u7b54\u6848)\s*[:\uff1a]\s*(.+)", raw, re.I | re.S)
    if final_match:
        final = final_match.group(1).strip().splitlines()[0].strip()
        if not disclosure_match:
            reasoning = raw[: final_match.start()].strip()
    else:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        final = lines[-1] if lines else ""

    reasoning = re.sub(
        r"^\s*(?:Reasoning|Reasoning Trace|\u63a8\u7406)\s*[:\uff1a]\s*",
        "",
        reasoning,
        flags=re.I,
    ).strip()
    return reasoning, disclosure, final


def safe_get(item: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default
