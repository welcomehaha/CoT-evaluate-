from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_a.io import iter_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample an annotation batch from annotation_queue.jsonl.")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260511)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(iter_jsonl(args.queue))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    selected = rows[: min(args.n, len(rows))]
    write_jsonl(args.out_jsonl, selected)
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "id",
            "source_dataset",
            "task_type",
            "question",
            "gold_answer",
            "hidden_cue",
            "cue_target",
            "needs_human_review",
            "answer_correct",
            "essential_propositions_final",
            "hidden_cue_valid",
            "cue_target_valid",
            "bias_direction_valid",
            "code_tests_safe",
            "notes",
        ]
        with Path(args.out_csv).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in selected:
                ann = row.get("annotation_fields", {})
                writer.writerow(
                    {
                        "id": row.get("id"),
                        "source_dataset": row.get("source_dataset"),
                        "task_type": row.get("task_type"),
                        "question": row.get("question"),
                        "gold_answer": row.get("gold_answer"),
                        "hidden_cue": row.get("hidden_cue"),
                        "cue_target": row.get("cue_target"),
                        "needs_human_review": "; ".join(row.get("needs_human_review", [])),
                        "answer_correct": ann.get("answer_correct"),
                        "essential_propositions_final": ann.get("essential_propositions_final"),
                        "hidden_cue_valid": ann.get("hidden_cue_valid"),
                        "cue_target_valid": ann.get("cue_target_valid"),
                        "bias_direction_valid": ann.get("bias_direction_valid"),
                        "code_tests_safe": ann.get("code_tests_safe"),
                        "notes": ann.get("notes", ""),
                    }
                )
    print(f"[annot] selected {len(selected)} items")


if __name__ == "__main__":
    main()
