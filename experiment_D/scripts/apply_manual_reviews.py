# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_d.io import iter_jsonl, write_jsonl  # noqa: E402

FILES = ["sft_train.jsonl", "rl_train.jsonl", "valid.jsonl", "mitigation_test.jsonl", "mitigation_suite.jsonl"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply manual reviews to Experiment D JSONL files.")
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clear-reviewed", action="store_true")
    return parser.parse_args()


def load_reviews(path: str) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = list(iter_jsonl(p))
    return {str(row.get("id") or row.get("sample_id")): row for row in rows if row.get("id") or row.get("sample_id")}


def main() -> None:
    args = parse_args()
    reviews = load_reviews(args.reviews)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for filename in FILES:
        path = Path(args.data_dir) / filename
        if not path.exists():
            continue
        rows = []
        updated = 0
        for item in iter_jsonl(path):
            review = reviews.get(str(item.get("id")))
            if review:
                metadata = dict(item.get("metadata") or {})
                metadata["manual_review"] = {k: review.get(k) for k in ["accuracy", "faithfulness_score", "used_hidden_cue", "verbalized_hidden_cue", "secrecy", "over_confession", "structured_fields_complete", "key_premise_recall", "user_satisfaction", "notes"] if k in review}
                metadata["human_reviewed"] = True
                item["metadata"] = metadata
                if review.get("notes"):
                    item["human_review_notes"] = str(review["notes"])
                if args.clear_reviewed:
                    item["needs_human_review"] = []
                updated += 1
            rows.append(item)
        write_jsonl(out_dir / filename, rows)
        total += updated
        print(f"[manual-D] {filename}: updated {updated}/{len(rows)}")
    print(f"[manual-D] total updated: {total}")


if __name__ == "__main__":
    main()
