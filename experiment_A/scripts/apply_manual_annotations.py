# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_a.io import iter_jsonl, write_jsonl  # noqa: E402


DATASET_FILES = [
    "sft_train.jsonl",
    "rl_train.jsonl",
    "valid.jsonl",
    "test_original.jsonl",
    "test_hidden_cue.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply manual annotations back to processed JSONL files.")
    parser.add_argument("--annotations", required=True, help="CSV or JSONL file produced from annotation_queue.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clear-reviewed", action="store_true")
    return parser.parse_args()


def parse_list_value(value: Any) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    return [text]


def load_annotations(path: str) -> dict[str, dict[str, Any]]:
    p = Path(path)
    rows: list[dict[str, Any]]
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = list(iter_jsonl(p))
    indexed = {}
    for row in rows:
        sample_id = row.get("id") or row.get("sample_id")
        if sample_id:
            indexed[str(sample_id)] = row
    return indexed


def apply_one(item: dict[str, Any], ann: dict[str, Any], clear_reviewed: bool) -> dict[str, Any]:
    updated = dict(item)
    props = parse_list_value(ann.get("essential_propositions_final") or ann.get("essential_propositions"))
    if props:
        updated["essential_propositions"] = props
    if ann.get("hidden_cue"):
        updated["hidden_cue"] = ann["hidden_cue"]
        updated["has_hidden_cue"] = True
    if ann.get("cue_target"):
        updated["cue_target"] = str(ann["cue_target"]).strip()
    metadata = dict(updated.get("metadata") or {})
    metadata["manual_annotation"] = {
        key: ann.get(key)
        for key in [
            "answer_correct",
            "hidden_cue_valid",
            "cue_target_valid",
            "bias_direction_valid",
            "code_tests_safe",
            "notes",
        ]
        if key in ann
    }
    metadata["human_validated"] = True
    updated["metadata"] = metadata
    if ann.get("notes"):
        updated["human_review_notes"] = str(ann["notes"])
    if clear_reviewed:
        updated["needs_human_review"] = []
    return updated


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotations)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_updated = 0
    for filename in DATASET_FILES:
        path = data_dir / filename
        if not path.exists():
            continue
        rows = []
        updated_count = 0
        for item in iter_jsonl(path):
            ann = annotations.get(str(item.get("id")))
            if ann:
                item = apply_one(item, ann, args.clear_reviewed)
                updated_count += 1
            rows.append(item)
        write_jsonl(out_dir / filename, rows)
        total_updated += updated_count
        print(f"[manual] {filename}: updated {updated_count}/{len(rows)}")
    print(f"[manual] total updated: {total_updated}")


if __name__ == "__main__":
    main()
