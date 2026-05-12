# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .intervention import add_review, as_original, with_hidden_cue
from .schema import CotSample
from .text import safe_get, split_rationale_to_props


def extract_boxed_answer(solution: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", solution or "")
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in str(solution).splitlines() if line.strip()]
    return lines[-1] if lines else ""


def parse_gsm8k_answer(answer: str) -> tuple[list[str], str]:
    if "####" in answer:
        rationale, final = answer.split("####", 1)
        final = final.strip()
    else:
        rationale, final = answer, answer
    return split_rationale_to_props(rationale), final


def _base_review(sample: CotSample) -> CotSample:
    add_review(sample, "essential_propositions_review: auto-extracted propositions are candidates and need human confirmation")
    return sample


def make_gsm8k(row: dict[str, Any], idx: int) -> CotSample:
    rationale, final = parse_gsm8k_answer(str(row.get("answer", "")))
    sample = CotSample(
        id=f"gsm8k_{idx:06d}",
        source_dataset="GSM8K",
        task_type="math",
        question=str(row.get("question", "")),
        choices=None,
        gold_answer=final,
        gold_rationale=rationale,
        essential_propositions=rationale,
        metadata={"raw_answer": row.get("answer", "")},
    )
    return _base_review(sample)


def make_math(row: dict[str, Any], idx: int) -> CotSample:
    solution = str(safe_get(row, "solution", "answer", default=""))
    props = split_rationale_to_props(solution, limit=10)
    sample = CotSample(
        id=f"math_{idx:06d}",
        source_dataset="MATH",
        task_type="math",
        question=str(safe_get(row, "problem", "question", default="")),
        choices=None,
        gold_answer=extract_boxed_answer(solution),
        gold_rationale=props,
        essential_propositions=props,
        metadata={"level": row.get("level"), "type": row.get("type"), "solution": solution},
    )
    return _base_review(sample)


def make_bbh(row: dict[str, Any], idx: int, config_name: str = "bbh") -> CotSample:
    target = str(safe_get(row, "target", "answer", default=""))
    sample = CotSample(
        id=f"bbh_{config_name}_{idx:06d}",
        source_dataset=f"BBH/{config_name}",
        task_type="logic",
        question=str(safe_get(row, "input", "question", "prompt", default="")),
        choices=None,
        gold_answer=target,
        gold_rationale=[],
        essential_propositions=[],
        metadata={"config": config_name},
    )
    add_review(sample, "essential_propositions_missing: BBH usually lacks gold CoT; add key premises manually or with a fixed evaluator")
    return sample


def make_humaneval(row: dict[str, Any], idx: int) -> CotSample:
    prompt = str(row.get("prompt", ""))
    tests = str(row.get("test", ""))
    sample = CotSample(
        id=f"humaneval_{idx:06d}",
        source_dataset="HumanEval",
        task_type="code",
        question=prompt,
        choices=None,
        gold_answer=str(row.get("canonical_solution", "")),
        gold_rationale=["Implement the requested Python function and satisfy the provided tests."],
        essential_propositions=["function signature must be preserved", "implementation must satisfy all unit tests"],
        metadata={"entry_point": row.get("entry_point"), "test": tests},
    )
    add_review(sample, "code_tests_review: execute tests only in a sandbox/container")
    add_review(sample, "essential_propositions_review: add task-specific edge cases from tests")
    return sample


def make_mbpp(row: dict[str, Any], idx: int) -> CotSample:
    tests = row.get("test_list", row.get("test_setup_code", []))
    sample = CotSample(
        id=f"mbpp_{idx:06d}",
        source_dataset="MBPP",
        task_type="code",
        question=str(safe_get(row, "text", "prompt", "question", default="")),
        choices=None,
        gold_answer=str(safe_get(row, "code", "solution", default="")),
        gold_rationale=["Write a Python solution that passes the supplied tests."],
        essential_propositions=["function behavior must match the natural language task", "solution must pass supplied tests"],
        metadata={"tests": tests},
    )
    add_review(sample, "code_tests_review: execute tests only in a sandbox/container")
    add_review(sample, "essential_propositions_review: add task-specific edge cases from tests")
    return sample


def make_bbq(row: dict[str, Any], idx: int) -> CotSample:
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
    full_question = f"{context}\n\n{question}".strip()
    sample = CotSample(
        id=f"bbq_{idx:06d}",
        source_dataset="BBQ",
        task_type="bias_choice",
        question=full_question,
        choices=choices,
        gold_answer=gold,
        gold_rationale=[],
        essential_propositions=[],
        metadata={"category": row.get("category"), "ambiguous": row.get("ambiguous"), "raw_label": label},
    )
    add_review(sample, "essential_propositions_missing: add unbiased decision rule and option elimination rationale")
    add_review(sample, "bias_direction_review: verify stereotype direction and ambiguous/disambiguated condition")
    return sample


def make_truthfulqa(row: dict[str, Any], idx: int) -> CotSample:
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
    sample = CotSample(
        id=f"truthfulqa_{idx:06d}",
        source_dataset="TruthfulQA",
        task_type="truthfulness",
        question=str(safe_get(row, "question", "prompt", default="")),
        choices=choices,
        gold_answer=gold,
        gold_rationale=[str(row.get("best_answer", ""))] if row.get("best_answer") else [],
        essential_propositions=[],
        metadata={"category": row.get("category")},
    )
    add_review(sample, "truthfulness_judge_review: open-ended or MC truthfulness needs fixed evaluator/human adjudication")
    add_review(sample, "essential_propositions_missing: add truth-preserving premises manually")
    return sample


def make_sycophancy(row: dict[str, Any], idx: int) -> CotSample:
    question = str(safe_get(row, "question", "prompt", "text", default=""))
    choices = None
    raw_choices = safe_get(row, "choices", "options", default=None)
    if isinstance(raw_choices, list):
        choices = {chr(65 + i): str(choice) for i, choice in enumerate(raw_choices)}
    gold = str(safe_get(row, "answer", "label", "correct_answer", default=""))
    sample = CotSample(
        id=f"sycophancy_{idx:06d}",
        source_dataset="SycophancyEval",
        task_type="sycophancy",
        question=question,
        choices=choices,
        gold_answer=gold,
        gold_rationale=[],
        essential_propositions=[],
        metadata={"raw_keys": sorted(row.keys())},
    )
    add_review(sample, "sycophancy_label_review: confirm the non-sycophantic answer and user-preference cue")
    add_review(sample, "essential_propositions_missing: add task-specific decision rule manually")
    return sample


def make_chinese_exam(row: dict[str, Any], idx: int, source: str) -> CotSample:
    choices = {key: str(row[key]) for key in ["A", "B", "C", "D"] if key in row}
    rationale = split_rationale_to_props(str(safe_get(row, "explanation", "analysis", default="")), limit=8)
    sample = CotSample(
        id=f"{source.lower()}_{idx:06d}",
        source_dataset=source,
        task_type="chinese_exam",
        question=str(safe_get(row, "question", "prompt", default="")),
        choices=choices or None,
        gold_answer=str(safe_get(row, "answer", "label", default="")),
        gold_rationale=rationale,
        essential_propositions=rationale,
        metadata={"subject": row.get("subject")},
    )
    add_review(sample, "essential_propositions_review: Chinese exam rationales often need manual key-premise extraction")
    return sample


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


def _sample(items: list[CotSample], n: int, rng: random.Random) -> list[CotSample]:
    rng.shuffle(items)
    return items[: min(n, len(items))]


def build_samples(recipe: dict[str, Any]) -> list[CotSample]:
    rng = random.Random(int(recipe.get("seed", 20260511)))
    specs = recipe.get("datasets", {})
    all_samples: list[CotSample] = []

    simple_builders: dict[str, Callable[[dict[str, Any], int], CotSample]] = {
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
            built: list[CotSample] = []
            configs = spec.get("configs") or [spec.get("hf_config")]
            for config_name in configs:
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

    for index, sample in enumerate(all_samples):
        sample.metadata["global_index"] = index
    return all_samples


def split_samples(samples: list[CotSample], recipe: dict[str, Any]) -> dict[str, list[CotSample]]:
    rng = random.Random(int(recipe.get("seed", 20260511)))
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
    valid = rl[:n_valid]
    rl_train = rl[n_valid:]
    return {
        "sft_train": sft,
        "rl_train": rl_train,
        "valid": valid,
        "test": test,
    }


def augment_train_with_cues(samples: list[CotSample], rng: random.Random, ratio: float = 0.35) -> list[CotSample]:
    augmented = [as_original(sample) for sample in samples]
    candidates = [sample for sample in samples if sample.choices]
    rng.shuffle(candidates)
    n_aug = int(len(candidates) * ratio)
    augmented.extend(with_hidden_cue(sample, rng) for sample in candidates[:n_aug])
    rng.shuffle(augmented)
    return augmented


def make_test_pairs(samples: list[CotSample], rng: random.Random) -> tuple[list[CotSample], list[CotSample]]:
    original = [as_original(sample) for sample in samples]
    hidden = [with_hidden_cue(sample, rng) for sample in samples]
    return original, hidden


def annotation_queue(*groups: list[CotSample]) -> list[dict[str, Any]]:
    queue = []
    seen: set[str] = set()
    for samples in groups:
        for sample in samples:
            if sample.needs_human_review and sample.id not in seen:
                seen.add(sample.id)
                item = sample.to_dict()
                item["annotation_fields"] = {
                    "answer_correct": None,
                    "essential_propositions_final": sample.essential_propositions,
                    "hidden_cue_valid": None,
                    "cue_target_valid": None,
                    "bias_direction_valid": None,
                    "code_tests_safe": None,
                    "notes": "",
                }
                queue.append(item)
    return queue


def save_manifest(path: str | Path, recipe: dict[str, Any], counts: dict[str, int]) -> None:
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps({"recipe": recipe, "counts": counts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
