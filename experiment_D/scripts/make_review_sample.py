# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_d.io import iter_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample manual review batch for Experiment D.")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(iter_jsonl(args.queue))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    selected = rows[: min(args.n, len(rows))]
    write_jsonl(args.out_jsonl, selected)
    if args.out_csv:
        fields = ["id", "source_dataset", "task_type", "difficulty", "risk", "has_hidden_cue", "hidden_cue", "cue_target", "task", "choices", "gold_answer", "needs_human_review", "accuracy", "faithfulness_score", "used_hidden_cue", "verbalized_hidden_cue", "secrecy", "over_confession", "structured_fields_complete", "key_premise_recall", "user_satisfaction", "notes"]
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out_csv).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in selected:
                review = row.get("review_fields", {})
                writer.writerow({"id": row.get("id"), "source_dataset": row.get("source_dataset"), "task_type": row.get("task_type"), "difficulty": row.get("difficulty"), "risk": row.get("risk"), "has_hidden_cue": row.get("has_hidden_cue"), "hidden_cue": row.get("hidden_cue"), "cue_target": row.get("cue_target"), "task": row.get("task"), "choices": row.get("choices"), "gold_answer": row.get("gold_answer"), "needs_human_review": "; ".join(row.get("needs_human_review", [])), **{key: review.get(key) for key in fields if key in review}})
    print(f"[review-D] selected {len(selected)} items")


if __name__ == "__main__":
    main()
