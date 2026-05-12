# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_d.io import iter_jsonl, write_jsonl  # noqa: E402
from experiment_d.metrics import aggregate, score_output, stratified, write_csv  # noqa: E402
from experiment_d.rewards import load_mitigation_config  # noqa: E402
from experiment_d.schema import MitigationSample, ModelOutput  # noqa: E402
from experiment_d.text import parse_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Experiment D mitigation outputs.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mitigation-config", default=str(ROOT / "configs" / "mitigation_config_D0_to_D5.yaml"))
    return parser.parse_args()


def tolerant_output(item: dict) -> ModelOutput:
    item = dict(item)
    item.setdefault("sample_id", item.get("id", item.get("sample_id")))
    item.setdefault("model", item.get("model_name", "unknown_model"))
    item.setdefault("group", item.get("group", item.get("model", "unknown_group")))
    if "raw_output" not in item:
        item["raw_output"] = "\n".join(str(item.get(key, "")) for key in ["reasoning", "cue_disclosure", "final_answer"] if item.get(key))
    if "reasoning" not in item or "cue_disclosure" not in item or "final_answer" not in item:
        reasoning, disclosure, final = parse_output(item["raw_output"])
        item.setdefault("reasoning", reasoning)
        item.setdefault("cue_disclosure", disclosure)
        item.setdefault("final_answer", final)
    return ModelOutput.from_dict({"sample_id": item["sample_id"], "model": item["model"], "group": item["group"], "raw_output": item["raw_output"], "reasoning": item["reasoning"], "cue_disclosure": item["cue_disclosure"], "final_answer": item["final_answer"], "output_tokens": item.get("output_tokens"), "input_tokens": item.get("input_tokens"), "latency_s": item.get("latency_s"), "metadata": item.get("metadata", {})})


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = {item["id"]: MitigationSample.from_dict(item) for item in iter_jsonl(args.samples)}
    outputs = []
    for path in args.outputs:
        outputs.extend(tolerant_output(item) for item in iter_jsonl(path))
    metrics = []
    for output in outputs:
        sample = samples.get(output.sample_id)
        if not sample:
            print(f"[warn] missing sample for {output.sample_id}")
            continue
        cfg, dyn_cfg, weights = load_mitigation_config(args.mitigation_config, output.group)
        metrics.append(score_output(sample, output, cfg, dyn_cfg, weights))
    rows = [m.to_dict() for m in metrics]
    summary = aggregate(metrics)
    strata = stratified(metrics)
    write_jsonl(out_dir / "per_sample_mitigation_metrics.jsonl", rows)
    write_csv(out_dir / "eval_mitigation_metrics.csv", summary)
    write_csv(out_dir / "stratified_metrics.csv", strata)
    (out_dir / "eval_mitigation_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval-D] wrote {len(rows)} per-sample metrics")
    print(f"[eval-D] wrote summary to {out_dir / 'eval_mitigation_metrics.csv'}")


if __name__ == "__main__":
    main()
