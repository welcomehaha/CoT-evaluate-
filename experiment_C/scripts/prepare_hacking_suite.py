# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_c.dataset_builders import audit_queue, build_tasks, save_manifest, split_tasks  # noqa: E402
from experiment_c.io import load_yaml, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Experiment C reward-hacking task suite.")
    parser.add_argument("--recipe", default=str(ROOT / "configs" / "dataset_pilot.yaml"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe = load_yaml(args.recipe)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(recipe)
    if not tasks:
        raise SystemExit("No tasks loaded. Check dataset names/configs or network access.")
    splits = split_tasks(tasks, recipe)
    files = {
        "sft_train": splits["sft_train"],
        "rl_train": splits["rl_train"],
        "valid": splits["valid"],
        "hacking_test": splits["test"],
        "hacking_task_suite": tasks,
    }
    for name, rows in files.items():
        write_jsonl(out_dir / f"{name}.jsonl", [task.to_dict() for task in rows])
        print(f"[prepare-C] wrote {name}: {len(rows)}")
    queue = audit_queue(splits["sft_train"], splits["rl_train"], splits["valid"], splits["test"])
    write_jsonl(out_dir / "human_audit_queue.jsonl", queue)
    counts = {name: len(rows) for name, rows in files.items()} | {"human_audit_queue": len(queue)}
    save_manifest(out_dir / "manifest.json", recipe, counts)
    print(f"[prepare-C] wrote human_audit_queue: {len(queue)}")


if __name__ == "__main__":
    main()
