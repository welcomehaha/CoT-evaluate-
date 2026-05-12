# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_a.io import iter_jsonl, write_jsonl  # noqa: E402
from experiment_a.metrics import aggregate_metrics, score_model_output, write_csv  # noqa: E402
from experiment_a.schema import CotSample, ModelOutput  # noqa: E402
from experiment_a.text import parse_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Experiment A model outputs.")
    parser.add_argument("--samples-original", required=True)
    parser.add_argument("--samples-hidden", required=True)
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def tolerant_output(item: dict) -> ModelOutput:
    if "raw_output" not in item:
        raw = "\n".join(
            str(item.get(key, ""))
            for key in ["reasoning", "reasoning_summary", "cue_disclosure", "final_answer"]
            if item.get(key)
        )
        item["raw_output"] = raw
    if "reasoning" not in item or "final_answer" not in item:
        reasoning, final = parse_output(item.get("raw_output", ""))
        item.setdefault("reasoning", reasoning)
        item.setdefault("final_answer", final)
    item.setdefault("model", item.get("model_name", "unknown_model"))
    item.setdefault("group", item.get("group", item.get("model", "unknown_group")))
    item.setdefault("prompt_version", item.get("prompt_version", "hidden_cue" if item.get("has_hidden_cue") else "original"))
    item.setdefault("sample_id", item.get("id", item.get("sample_id")))
    return ModelOutput.from_dict(
        {
            "sample_id": item["sample_id"],
            "model": item["model"],
            "group": item["group"],
            "prompt_version": item["prompt_version"],
            "raw_output": item["raw_output"],
            "reasoning": item["reasoning"],
            "final_answer": item["final_answer"],
            "output_tokens": item.get("output_tokens"),
            "input_tokens": item.get("input_tokens"),
            "latency_s": item.get("latency_s"),
            "metadata": item.get("metadata", {}),
        }
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_samples = {item["id"]: CotSample.from_dict(item) for item in iter_jsonl(args.samples_original)}
    hidden_samples = {item["id"]: CotSample.from_dict(item) for item in iter_jsonl(args.samples_hidden)}

    outputs: list[ModelOutput] = []
    for path in args.outputs:
        outputs.extend(tolerant_output(item) for item in iter_jsonl(path))

    original_outputs: dict[tuple[str, str, str], ModelOutput] = {}
    for out in outputs:
        if out.prompt_version == "original":
            original_outputs[(out.model, out.group, out.sample_id)] = out

    per_sample = []
    for out in outputs:
        sample = hidden_samples[out.sample_id] if out.prompt_version == "hidden_cue" else original_samples[out.sample_id]
        original = original_outputs.get((out.model, out.group, out.sample_id))
        metric = score_model_output(sample, out, original_output=original)
        per_sample.append(metric)

    per_rows = [metric.to_dict() for metric in per_sample]
    summary = aggregate_metrics(per_sample)
    write_jsonl(out_dir / "per_sample_metrics.jsonl", per_rows)
    write_csv(out_dir / "summary.csv", summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] wrote {len(per_rows)} per-sample metrics")
    print(f"[eval] wrote summary to {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
