# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_a.dataset_builders import (  # noqa: E402
    annotation_queue,
    augment_train_with_cues,
    build_samples,
    make_test_pairs,
    save_manifest,
    split_samples,
)
from experiment_a.io import write_jsonl  # noqa: E402
from experiment_a.text import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Experiment A datasets.")
    parser.add_argument("--recipe", default=str(ROOT / "configs" / "dataset_pilot.yaml"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--sft-cue-ratio", type=float, default=0.35)
    parser.add_argument("--rl-cue-ratio", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe = load_yaml(args.recipe)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(recipe.get("seed", 20260511)))

    print(f"[prepare] loading and converting datasets from {args.recipe}")
    samples = build_samples(recipe)
    if not samples:
        raise SystemExit(
            "No samples were loaded. Check network access, Hugging Face dataset names, or set enabled=false for unavailable datasets."
        )
    splits = split_samples(samples, recipe)

    sft_train = augment_train_with_cues(splits["sft_train"], rng, ratio=args.sft_cue_ratio)
    rl_train = augment_train_with_cues(splits["rl_train"], rng, ratio=args.rl_cue_ratio)
    valid = augment_train_with_cues(splits["valid"], rng, ratio=args.rl_cue_ratio)
    test_original, test_hidden = make_test_pairs(splits["test"], rng)

    outputs = {
        "sft_train": sft_train,
        "rl_train": rl_train,
        "valid": valid,
        "test_original": test_original,
        "test_hidden_cue": test_hidden,
    }
    for name, items in outputs.items():
        write_jsonl(out_dir / f"{name}.jsonl", [sample.to_dict() for sample in items])
        print(f"[prepare] wrote {name}: {len(items)}")

    queue = annotation_queue(sft_train, rl_train, valid, test_original, test_hidden)
    write_jsonl(out_dir / "annotation_queue.jsonl", queue)
    print(f"[prepare] wrote annotation_queue: {len(queue)}")

    counts = {name: len(items) for name, items in outputs.items()}
    counts["annotation_queue"] = len(queue)
    save_manifest(out_dir / "manifest.json", recipe, counts)
    print(f"[prepare] manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
