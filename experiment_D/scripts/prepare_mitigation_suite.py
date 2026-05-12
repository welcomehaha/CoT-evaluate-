# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_d.dataset_builders import build_samples, review_queue, save_manifest, split_samples  # noqa: E402
from experiment_d.io import load_yaml, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Experiment D mitigation suite.")
    parser.add_argument("--recipe", default=str(ROOT / "configs" / "dataset_pilot.yaml"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe = load_yaml(args.recipe)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = build_samples(recipe)
    if not samples:
        raise SystemExit("No samples loaded. Check dataset names/configs or network access.")
    splits = split_samples(samples, recipe)
    files = {
        "sft_train": splits["sft_train"],
        "rl_train": splits["rl_train"],
        "valid": splits["valid"],
        "mitigation_test": splits["test"],
        "mitigation_suite": samples,
    }
    for name, rows in files.items():
        write_jsonl(out_dir / f"{name}.jsonl", [x.to_dict() for x in rows])
        print(f"[prepare-D] wrote {name}: {len(rows)}")
    queue = review_queue(splits["sft_train"], splits["rl_train"], splits["valid"], splits["test"])
    write_jsonl(out_dir / "human_review_queue.jsonl", queue)
    counts = {name: len(rows) for name, rows in files.items()} | {"human_review_queue": len(queue)}
    save_manifest(out_dir / "manifest.json", recipe, counts)
    print(f"[prepare-D] wrote human_review_queue: {len(queue)}")


if __name__ == "__main__":
    main()
