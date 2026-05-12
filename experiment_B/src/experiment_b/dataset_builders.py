# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .cues import add_review, as_original, choose_relation, with_cue
from .schema import HiddenCueSample
from .text import safe_get, split_rationale_to_props


def parse_gsm8k_answer(answer: str) -> tuple[list[str], str]:
    if "####" in answer:
        rationale, final = answer.split("####", 1)
    else:
        rationale, final = answer, answer
    return split_rationale_to_props(rationale), final.strip()


def extract_boxed_answer(solution: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", solution or "")
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in str(solution).splitlines() if line.strip()]
    return lines[-1] if lines else ""


def base_sample(
    sample_id: str,
    source_dataset: str,
    task_type: str,
    question: str,
    choices: dict[str, str] | None,
    gold_answer: str,
    gold_rationale: list[str] | None = None,
    essential_propositions: list[str] | None = None,
    language: str = "en",
    metadata: dict[str, Any] | None = None,
) -> HiddenCueSample:
    sample = HiddenCueSample(
        id=sample_id,
        base_sample_id=sample_id,
        variant_id="base",
        source_dataset=source_dataset,
        task_type=task_type,
        question=question,
        choices=choices,
        gold_answer=gold_answer,
        gold_rationale=gold_rationale or [],
        essential_propositions=essential_propositions or [],
        language=language,
        metadata=metadata or {},
    )
    if not sample.essential_propositions:
        add_review(sample, "essential_propositions_missing: add key premises/decision rule manually or with fixed evaluator")
    else:
        add_review(sample, "essential_propositions_review: auto-extracted propositions are candidates and need confirmation")
    return sample


def make_gsm8k(row: dict[str, Any], idx: int) -> HiddenCueSample:
    props, final = parse_gsm8k_answer(str(row.get("answer", "")))
    return base_sample(
        f"gsm8k_{idx:06d}",
        "GSM8K",
        "math",
        str(row.get("question", "")),
        None,
        final,
        props,
        props,
        metadata={"raw_answer": row.get("answer")},
    )


def make_math(row: dict[str, Any], idx: int) -> HiddenCueSample:
    solution = str(safe_get(row, "solution", "answer", default=""))
    props = split_rationale_to_props(solution, limit=10)
    return base_sample(
        f"math_{idx:06d}",
        "MATH",
        "math",
        str(safe_get(row, "problem", "question", default="")),
        None,
        extract_boxed_answer(solution),
        props,
        props,
        metadata={"level": row.get("level"), "type": row.get("type")},
    )


def make_bbh(row: dict[str, Any], idx: int, config_name: str = "bbh") -> HiddenCueSample:
    return base_sample(
        f"bbh_{config_name}_{idx:06d}",
        f"BBH/{config_name}",
        "choice_reasoning",
        str(safe_get(row, "input", "question", "prompt", default="")),
        None,
        str(safe_get(row, "target", "answer", default="")),
        [],
        [],
        metadata={"config": config_name},
    )


def make_humaneval(row: dict[str, Any], idx: int) -> HiddenCueSample:
    sample = base_sample(
        f"humaneval_{idx:06d}",
        "HumanEval",
        "code",
        str(row.get("prompt", "")),
        None,
        str(row.get("canonical_solution", "")),
        ["Implement the requested Python function and satisfy the provided tests."],
        ["function signature must be preserved", "implementation must satisfy all unit tests"],
        metadata={"entry_point": row.get("entry_point"), "test": row.get("test")},
    )
    add_review(sample, "code_tests_review: execute tests only in isolated sandbox/container")
    add_review(sample, "reward_shortcut_review: shortcut cue has no direct answer target; judge rationalization manually")
    return sample


def make_mbpp(row: dict[str, Any], idx: int) -> HiddenCueSample:
    sample = base_sample(
        f"mbpp_{idx:06d}",
        "MBPP",
        "code",
        str(safe_get(row, "text", "prompt", "question", default="")),
        None,
        str(safe_get(row, "code", "solution", default="")),
        ["Write a Python solution that passes the supplied tests."],
        ["function behavior must match the natural language task", "solution must pass supplied tests"],
        metadata={"tests": row.get("test_list", row.get("test_setup_code", []))},
    )
    add_review(sample, "code_tests_review: execute tests only in isolated sandbox/container")
    add_review(sample, "reward_shortcut_review: shortcut cue has no direct answer target; judge rationalization manually")
    return sample


def make_bbq(row: dict[str, Any], idx: int) -> HiddenCueSample:
    choices = {
        "A": str(safe_get(row, "ans0", "answerA", "A", default="")),
        "B": str(safe_get(row, "ans1", "answerB", "B", default="")),
        "C": str(safe_get(row, "ans2", "answerC", "C", default="")),
    }
    label = safe_get(row, "label", "answer", default=0)
    if isinstance(label, str) and label.strip().upper() in choices:
        gold = label.strip().upper()
    else:
        try:
            gold = ["A", "B", "C"][int(label)]
        except Exception:
            gold = str(label)
    context = str(row.get("context", ""))
    question = str(row.get("question", ""))
    sample = base_sample(
        f"bbq_{idx:06d}",
        "BBQ",
        "bias_choice",
        f"{context}\n\n{question}".strip(),
        choices,
        gold,
        [],
        [],
        metadata={"category": row.get("category"), "ambiguous": row.get("ambiguous"), "raw_label": label},
    )
    add_review(sample, "bias_direction_review: verify stereotype/majority cue direction")
    return sample


def make_truthfulqa(row: dict[str, Any], idx: int) -> HiddenCueSample:
    choices: dict[str, str] | None = None
    gold = ""
    mc_targets = row.get("mc1_targets") or row.get("mc2_targets")
    if isinstance(mc_targets, dict):
        raw_choices = mc_targets.get("choices", [])
        labels = mc_targets.get("labels", [])
        choices = {chr(65 + i): str(choice) for i, choice in enumerate(raw_choices)}
        for i, label in enumerate(labels):
            if int(label) == 1:
                gold = chr(65 + i)
                break
    if not gold:
        gold = str(safe_get(row, "best_answer", "correct_answer", "answer", default=""))
    sample = base_sample(
        f"truthfulqa_{idx:06d}",
        "TruthfulQA",
        "truthfulness",
        str(safe_get(row, "question", "prompt", default="")),
        choices,
        gold,
        [str(row.get("best_answer", ""))] if row.get("best_answer") else [],
        [],
        metadata={"category": row.get("category")},
    )
    add_review(sample, "common_misconception_review: verify cue target is a common misconception")
    return sample


def make_sycophancy(row: dict[str, Any], idx: int) -> HiddenCueSample:
    raw_choices = safe_get(row, "choices", "options", default=None)
    choices = {chr(65 + i): str(choice) for i, choice in enumerate(raw_choices)} if isinstance(raw_choices, list) else None
    sample = base_sample(
        f"sycophancy_{idx:06d}",
        "SycophancyEval",
        "sycophancy",
        str(safe_get(row, "question", "prompt", "text", default="")),
        choices,
        str(safe_get(row, "answer", "label", "correct_answer", default="")),
        [],
        [],
        metadata={"raw_keys": sorted(row.keys())},
    )
    add_review(sample, "sycophancy_label_review: confirm non-sycophantic answer and conflicting user preference")
    return sample


def make_chinese_exam(row: dict[str, Any], idx: int, source: str) -> HiddenCueSample:
    choices = {key: str(row[key]) for key in ["A", "B", "C", "D"] if key in row}
    rationale = split_rationale_to_props(str(safe_get(row, "explanation", "analysis", default="")), limit=8)
    return base_sample(
        f"{source.lower()}_{idx:06d}",
        source,
        "chinese_exam",
        str(safe_get(row, "question", "prompt", default="")),
        choices or None,
        str(safe_get(row, "answer", "label", default="")),
        rationale,
        rationale,
        language="zh",
        metadata={"subject": row.get("subject")},
    )


def load_rows(hf_name: str, hf_config: str | None, splits: list[str]) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    for split in splits:
        try:
            ds = load_dataset(hf_name, hf_config, split=split) if hf_config else load_dataset(hf_name, split=split)
            rows.extend(dict(item) for item in ds)
        except Exception as exc:
            print(f"[warn] could not load {hf_name} config={hf_config} split={split}: {exc}")
    return rows


def _sample(items: list[HiddenCueSample], n: int, rng: random.Random) -> list[HiddenCueSample]:
    rng.shuffle(items)
    return items[: min(n, len(items))]


def build_base_samples(recipe: dict[str, Any]) -> list[HiddenCueSample]:
    rng = random.Random(int(recipe.get("seed", 20260512)))
    specs = recipe.get("datasets", {})
    all_samples: list[HiddenCueSample] = []
    simple_builders: dict[str, Callable[[dict[str, Any], int], HiddenCueSample]] = {
        "gsm8k": make_gsm8k,
        "math": make_math,
        "humaneval": make_humaneval,
        "mbpp": make_mbpp,
        "bbq": make_bbq,
        "truthfulqa": make_truthfulqa,
        "sycophancy": make_sycophancy,
    }

    for name, spec in specs.items():
        if not spec.get("enabled", False):
            continue
        target_count = int(spec.get("target_count", 0))
        if target_count <= 0:
            continue
        if name == "bbh":
            built: list[HiddenCueSample] = []
            for config_name in spec.get("configs") or [spec.get("hf_config")]:
                rows = load_rows(spec["hf_name"], config_name, spec.get("splits", ["test"]))
                built.extend(make_bbh(row, idx, str(config_name)) for idx, row in enumerate(rows))
            all_samples.extend(_sample(built, target_count, rng))
            continue
        if name in {"ceval", "cmmlu"}:
            rows = load_rows(spec["hf_name"], spec.get("hf_config"), spec.get("splits", ["test"]))
            built = [make_chinese_exam(row, idx, name.upper()) for idx, row in enumerate(rows)]
            all_samples.extend(_sample(built, target_count, rng))
            continue
        builder = simple_builders.get(name)
        if builder is None:
            print(f"[warn] no builder for dataset key {name}")
            continue
        rows = load_rows(spec["hf_name"], spec.get("hf_config"), spec.get("splits", ["train"]))
        built = [builder(row, idx) for idx, row in enumerate(rows)]
        all_samples.extend(_sample(built, target_count, rng))

    for i, sample in enumerate(all_samples):
        sample.metadata["global_index"] = i
    return all_samples


def split_samples(samples: list[HiddenCueSample], recipe: dict[str, Any]) -> dict[str, list[HiddenCueSample]]:
    rng = random.Random(int(recipe.get("seed", 20260512)))
    items = list(samples)
    rng.shuffle(items)
    split_cfg = recipe.get("splits", {"sft_train": 0.7, "rl_train": 0.2, "test": 0.1})
    n = len(items)
    n_sft = int(n * float(split_cfg.get("sft_train", 0.7)))
    n_rl = int(n * float(split_cfg.get("rl_train", 0.2)))
    sft = items[:n_sft]
    rl = items[n_sft : n_sft + n_rl]
    test = items[n_sft + n_rl :]
    valid_ratio = float(recipe.get("valid_from_rl_ratio", 0.1))
    n_valid = max(1, int(len(rl) * valid_ratio)) if rl else 0
    return {"sft_train": sft, "rl_train": rl[n_valid:], "valid": rl[:n_valid], "test": test}


def build_cue_variants(
    samples: list[HiddenCueSample],
    recipe: dict[str, Any],
    split: str,
    training: bool,
) -> list[HiddenCueSample]:
    cue_cfg = recipe.get("cue_generation", {})
    cue_types = cue_cfg.get("cue_types", ["authority", "majority", "user_preference"])
    weights = cue_cfg.get("relation_weights", {"conflict": 0.7, "consistent": 0.2, "neutral": 0.1})
    split_offset = sum(ord(ch) for ch in split)
    rng = random.Random(int(recipe.get("seed", 20260512)) + split_offset)

    variants: list[HiddenCueSample] = [as_original(sample, split=split) for sample in samples]
    if training:
        ratio = float(cue_cfg.get("train_cue_ratio", 0.7))
        candidates = list(samples)
        rng.shuffle(candidates)
        selected = candidates[: int(len(candidates) * ratio)]
        for idx, sample in enumerate(selected):
            cue_type = rng.choice(cue_types)
            relation = "neutral" if cue_type == "reward_shortcut" else choose_relation(weights, rng, True)
            variants.append(with_cue(sample, cue_type, relation, rng, idx, split=split))
        rng.shuffle(variants)
        return variants

    cues_per_sample = int(cue_cfg.get("cues_per_test_sample", 3))
    for sample in samples:
        chosen_types = list(cue_types)
        rng.shuffle(chosen_types)
        for idx, cue_type in enumerate(chosen_types[:cues_per_sample]):
            relation = "neutral" if cue_type == "reward_shortcut" else choose_relation(weights, rng, True)
            variants.append(with_cue(sample, cue_type, relation, rng, idx, split=split))
    return variants


def split_test_variants(variants: list[HiddenCueSample]) -> tuple[list[HiddenCueSample], list[HiddenCueSample]]:
    original = [sample for sample in variants if sample.prompt_version == "original"]
    cue = [sample for sample in variants if sample.prompt_version == "cue"]
    return original, cue


def annotation_queue(*groups: list[HiddenCueSample]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    seen: set[str] = set()
    for samples in groups:
        for sample in samples:
            if sample.needs_human_review and sample.id not in seen:
                seen.add(sample.id)
                item = sample.to_dict()
                item["annotation_fields"] = {
                    "cue_valid": None,
                    "cue_target_valid": None,
                    "cue_relation_valid": None,
                    "essential_propositions_final": sample.essential_propositions,
                    "used_hidden_cue": None,
                    "verbalized_hidden_cue": None,
                    "rationalization": None,
                    "explanation_quality": None,
                    "faithfulness_risk": None,
                    "notes": "",
                }
                queue.append(item)
    return queue


def save_manifest(path: str | Path, recipe: dict[str, Any], counts: dict[str, int]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"recipe": recipe, "counts": counts}, ensure_ascii=False, indent=2), encoding="utf-8")
