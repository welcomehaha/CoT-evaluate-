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

from experiment_c.io import iter_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample manual audit batch for Experiment C.")
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
        fields = ["id", "source_dataset", "task_type", "shortcut_type", "task", "gold_output", "shortcut_opportunity", "forbidden_actions", "needs_human_audit", "task_success", "hack_detected", "hack_type", "shortcut_used", "shortcut_disclosed", "behavioral_failure", "monitor_detected", "faithfulness_risk", "notes"]
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out_csv).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in selected:
                audit = row.get("audit_fields", {})
                writer.writerow(
                    {
                        "id": row.get("id"),
                        "source_dataset": row.get("source_dataset"),
                        "task_type": row.get("task_type"),
                        "shortcut_type": row.get("shortcut_type"),
                        "task": row.get("task"),
                        "gold_output": row.get("gold_output"),
                        "shortcut_opportunity": row.get("shortcut_opportunity"),
                        "forbidden_actions": "; ".join(row.get("forbidden_actions", [])),
                        "needs_human_audit": "; ".join(row.get("needs_human_audit", [])),
                        **{key: audit.get(key) for key in fields if key in audit},
                    }
                )
    print(f"[audit-C] selected {len(selected)} items")


if __name__ == "__main__":
    main()
