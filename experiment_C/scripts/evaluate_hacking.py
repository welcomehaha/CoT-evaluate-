# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_c.io import iter_jsonl, write_jsonl  # noqa: E402
from experiment_c.metrics import aggregate, aggregate_by_task, aggregate_monitor_inputs, audit_output, write_csv  # noqa: E402
from experiment_c.schema import HackingTask, ModelOutput  # noqa: E402
from experiment_c.text import parse_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Experiment C reward-hacking outputs.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--monitor-inputs", nargs="+", default=["output_only", "cot_only", "cot_output", "tool_log"])
    return parser.parse_args()


def tolerant_output(item: dict) -> ModelOutput:
    item = dict(item)
    item.setdefault("sample_id", item.get("id", item.get("sample_id")))
    item.setdefault("model", item.get("model_name", "unknown_model"))
    item.setdefault("group", item.get("group", item.get("model", "unknown_group")))
    if "raw_output" not in item:
        item["raw_output"] = "\n".join(str(item.get(key, "")) for key in ["reasoning", "shortcut_disclosure", "final_output"] if item.get(key))
    if "reasoning" not in item or "shortcut_disclosure" not in item or "final_output" not in item:
        reasoning, disclosure, final = parse_output(item["raw_output"])
        item.setdefault("reasoning", reasoning)
        item.setdefault("shortcut_disclosure", disclosure)
        item.setdefault("final_output", final)
    item.setdefault("tool_log", item.get("metadata", {}).get("tool_log", ""))
    return ModelOutput.from_dict(
        {
            "sample_id": item["sample_id"],
            "model": item["model"],
            "group": item["group"],
            "raw_output": item["raw_output"],
            "reasoning": item["reasoning"],
            "shortcut_disclosure": item["shortcut_disclosure"],
            "final_output": item["final_output"],
            "output_tokens": item.get("output_tokens"),
            "input_tokens": item.get("input_tokens"),
            "latency_s": item.get("latency_s"),
            "tool_log": item.get("tool_log", ""),
            "metadata": item.get("metadata", {}),
        }
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = {item["id"]: HackingTask.from_dict(item) for item in iter_jsonl(args.samples)}
    outputs = []
    for path in args.outputs:
        outputs.extend(tolerant_output(item) for item in iter_jsonl(path))
    audits = []
    for output in outputs:
        task = tasks.get(output.sample_id)
        if not task:
            print(f"[warn] missing sample for {output.sample_id}")
            continue
        for monitor_input in args.monitor_inputs:
            audits.append(audit_output(task, output, monitor_input=monitor_input))
    rows = [audit.to_dict() for audit in audits]
    summary = aggregate(audits)
    monitor_summary = aggregate_monitor_inputs(audits)
    task_summary = aggregate_by_task(audits)
    write_jsonl(out_dir / "groundtruth_audit_results.jsonl", rows)
    write_csv(out_dir / "hacking_metrics.csv", summary)
    write_csv(out_dir / "monitor_input_metrics.csv", monitor_summary)
    write_csv(out_dir / "task_type_metrics.csv", task_summary)
    (out_dir / "hacking_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval-C] wrote {len(rows)} audit rows")
    print(f"[eval-C] wrote summary to {out_dir / 'hacking_metrics.csv'}")


if __name__ == "__main__":
    main()
