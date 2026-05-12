# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_c.io import iter_jsonl, write_jsonl  # noqa: E402


DATASET_FILES = ["sft_train.jsonl", "rl_train.jsonl", "valid.jsonl", "hacking_test.jsonl", "hacking_task_suite.jsonl"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply manual audit annotations to Experiment C task files.")
    parser.add_argument("--audits", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clear-reviewed", action="store_true")
    return parser.parse_args()


def load_audits(path: str) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = list(iter_jsonl(p))
    return {str(row.get("id") or row.get("sample_id")): row for row in rows if row.get("id") or row.get("sample_id")}


def apply_one(item: dict[str, Any], audit: dict[str, Any], clear_reviewed: bool) -> dict[str, Any]:
    updated = dict(item)
    metadata = dict(updated.get("metadata") or {})
    metadata["manual_audit"] = {
        key: audit.get(key)
        for key in [
            "task_success",
            "hack_detected",
            "hack_type",
            "shortcut_used",
            "shortcut_disclosed",
            "behavioral_failure",
            "monitor_detected",
            "faithfulness_risk",
            "notes",
        ]
        if key in audit
    }
    metadata["human_audited"] = True
    updated["metadata"] = metadata
    if audit.get("notes"):
        updated["human_audit_notes"] = str(audit["notes"])
    if clear_reviewed:
        updated["needs_human_audit"] = []
    return updated


def main() -> None:
    args = parse_args()
    audits = load_audits(args.audits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for filename in DATASET_FILES:
        path = Path(args.data_dir) / filename
        if not path.exists():
            continue
        rows = []
        updated_count = 0
        for item in iter_jsonl(path):
            audit = audits.get(str(item.get("id")))
            if audit:
                item = apply_one(item, audit, args.clear_reviewed)
                updated_count += 1
            rows.append(item)
        write_jsonl(out_dir / filename, rows)
        total += updated_count
        print(f"[manual-C] {filename}: updated {updated_count}/{len(rows)}")
    print(f"[manual-C] total updated: {total}")


if __name__ == "__main__":
    main()
