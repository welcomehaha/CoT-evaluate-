# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_b.io import iter_jsonl, write_jsonl  # noqa: E402
from experiment_b.metrics import aggregate, aggregate_by_cue_type, mcnemar_rows, original_key, pair_score, write_csv  # noqa: E402
from experiment_b.schema import HiddenCueSample, ModelOutput  # noqa: E402
from experiment_b.text import parse_model_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Experiment B hidden-cue outputs.")
    parser.add_argument("--samples-original", required=True)
    parser.add_argument("--samples-cue", required=True)
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def tolerant_output(item: dict) -> ModelOutput:
    item = dict(item)
    item.setdefault("sample_id", item.get("id", item.get("sample_id")))
    item.setdefault("base_sample_id", item.get("base_sample_id", item["sample_id"].split("__")[0]))
    item.setdefault("variant_id", item.get("variant_id", "original" if item.get("prompt_version") == "original" else "unknown"))
    item.setdefault("model", item.get("model_name", "unknown_model"))
    item.setdefault("group", item.get("group", item.get("model", "unknown_group")))
    item.setdefault("prompt_version", item.get("prompt_version", "cue" if "__" in item["sample_id"] else "original"))
    if "raw_output" not in item:
        item["raw_output"] = "\n".join(
            str(item.get(key, ""))
            for key in ["reasoning_summary", "reasoning", "cue_disclosure", "final_answer"]
            if item.get(key)
        )
    if "reasoning_summary" not in item or "cue_disclosure" not in item or "final_answer" not in item:
        reasoning, disclosure, final = parse_model_output(item["raw_output"])
        item.setdefault("reasoning_summary", reasoning)
        item.setdefault("cue_disclosure", disclosure)
        item.setdefault("final_answer", final)
    return ModelOutput.from_dict(
        {
            "sample_id": item["sample_id"],
            "base_sample_id": item["base_sample_id"],
            "variant_id": item["variant_id"],
            "model": item["model"],
            "group": item["group"],
            "prompt_version": item["prompt_version"],
            "raw_output": item["raw_output"],
            "reasoning_summary": item["reasoning_summary"],
            "cue_disclosure": item["cue_disclosure"],
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

    original_samples = {item["base_sample_id"]: HiddenCueSample.from_dict(item) for item in iter_jsonl(args.samples_original)}
    cue_samples = {item["id"]: HiddenCueSample.from_dict(item) for item in iter_jsonl(args.samples_cue)}

    outputs: list[ModelOutput] = []
    for path in args.outputs:
        outputs.extend(tolerant_output(item) for item in iter_jsonl(path))

    original_outputs: dict[tuple[str, str, str], ModelOutput] = {}
    cue_outputs: list[ModelOutput] = []
    for output in outputs:
        if output.prompt_version == "original":
            original_outputs[original_key(output)] = output
        else:
            cue_outputs.append(output)

    metrics = []
    for output in cue_outputs:
        sample = cue_samples.get(output.sample_id)
        if sample is None:
            print(f"[warn] missing cue sample for output {output.sample_id}")
            continue
        original_sample = original_samples.get(output.base_sample_id)
        original_output = original_outputs.get((output.model, output.group, output.base_sample_id))
        metrics.append(pair_score(sample, output, original_sample=original_sample, original_output=original_output))

    per_rows = [metric.to_dict() for metric in metrics]
    summary = aggregate(metrics)
    cue_type_summary = aggregate_by_cue_type(metrics)
    mcnemar = mcnemar_rows(metrics)

    write_jsonl(out_dir / "per_sample_hidden_cue_metrics.jsonl", per_rows)
    write_csv(out_dir / "eval_hidden_cue_metrics.csv", summary)
    write_csv(out_dir / "cue_type_metrics.csv", cue_type_summary)
    write_csv(out_dir / "mcnemar_by_group.csv", mcnemar)
    (out_dir / "eval_hidden_cue_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval-B] wrote {len(per_rows)} per-sample metrics")
    print(f"[eval-B] wrote summary to {out_dir / 'eval_hidden_cue_metrics.csv'}")


if __name__ == "__main__":
    main()
