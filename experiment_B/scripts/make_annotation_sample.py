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

from experiment_b.io import iter_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample manual annotation batch for Experiment B.")
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
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "id",
            "base_sample_id",
            "source_dataset",
            "task_type",
            "prompt_version",
            "cue_type",
            "cue_relation",
            "cue_target",
            "question",
            "choices",
            "gold_answer",
            "hidden_cue",
            "needs_human_review",
            "cue_valid",
            "cue_target_valid",
            "cue_relation_valid",
            "essential_propositions_final",
            "used_hidden_cue",
            "verbalized_hidden_cue",
            "rationalization",
            "explanation_quality",
            "faithfulness_risk",
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
                        "base_sample_id": row.get("base_sample_id"),
                        "source_dataset": row.get("source_dataset"),
                        "task_type": row.get("task_type"),
                        "prompt_version": row.get("prompt_version"),
                        "cue_type": row.get("cue_type"),
                        "cue_relation": row.get("cue_relation"),
                        "cue_target": row.get("cue_target"),
                        "question": row.get("question"),
                        "choices": row.get("choices"),
                        "gold_answer": row.get("gold_answer"),
                        "hidden_cue": row.get("hidden_cue"),
                        "needs_human_review": "; ".join(row.get("needs_human_review", [])),
                        "cue_valid": ann.get("cue_valid"),
                        "cue_target_valid": ann.get("cue_target_valid"),
                        "cue_relation_valid": ann.get("cue_relation_valid"),
                        "essential_propositions_final": ann.get("essential_propositions_final"),
                        "used_hidden_cue": ann.get("used_hidden_cue"),
                        "verbalized_hidden_cue": ann.get("verbalized_hidden_cue"),
                        "rationalization": ann.get("rationalization"),
                        "explanation_quality": ann.get("explanation_quality"),
                        "faithfulness_risk": ann.get("faithfulness_risk"),
                        "notes": ann.get("notes", ""),
                    }
                )
    print(f"[annot-B] selected {len(selected)} items")


if __name__ == "__main__":
    main()
