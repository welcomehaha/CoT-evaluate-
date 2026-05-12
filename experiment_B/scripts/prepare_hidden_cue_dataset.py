# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_b.dataset_builders import (  # noqa: E402
    annotation_queue,
    build_base_samples,
    build_cue_variants,
    save_manifest,
    split_samples,
    split_test_variants,
)
from experiment_b.io import load_yaml, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Experiment B hidden-cue datasets.")
    parser.add_argument("--recipe", default=str(ROOT / "configs" / "dataset_pilot.yaml"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe = load_yaml(args.recipe)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare-B] loading base datasets from {args.recipe}")
    base_samples = build_base_samples(recipe)
    if not base_samples:
        raise SystemExit("No samples loaded. Check network access and dataset names/configs.")

    splits = split_samples(base_samples, recipe)
    sft_train = build_cue_variants(splits["sft_train"], recipe, "sft_train", training=True)
    rl_train = build_cue_variants(splits["rl_train"], recipe, "rl_train", training=True)
    valid = build_cue_variants(splits["valid"], recipe, "valid", training=True)
    test_variants = build_cue_variants(splits["test"], recipe, "test", training=False)
    test_original, test_cue = split_test_variants(test_variants)

    files = {
        "sft_train": sft_train,
        "rl_train": rl_train,
        "valid": valid,
        "test_original": test_original,
        "test_cue": test_cue,
        "processed_hidden_cue_test": test_variants,
    }
    for name, samples in files.items():
        write_jsonl(out_dir / f"{name}.jsonl", [sample.to_dict() for sample in samples])
        print(f"[prepare-B] wrote {name}: {len(samples)}")

    queue = annotation_queue(sft_train, rl_train, valid, test_variants)
    write_jsonl(out_dir / "human_annotation_queue.jsonl", queue)
    print(f"[prepare-B] wrote human_annotation_queue: {len(queue)}")

    counts = {name: len(samples) for name, samples in files.items()}
    counts["human_annotation_queue"] = len(queue)
    save_manifest(out_dir / "manifest.json", recipe, counts)
    print(f"[prepare-B] manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
