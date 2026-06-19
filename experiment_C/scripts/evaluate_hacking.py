# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from experiment_c.io import iter_jsonl, write_jsonl  # noqa: E402
from experiment_c.metrics import (  # noqa: E402
    aggregate,
    aggregate_by_task,
    aggregate_monitor_inputs,
    aggregate_with_ci,
    audit_output,
    pairwise_tests,
    paper_table,
    write_csv,
)
from experiment_c.schema import HackingTask, ModelOutput  # noqa: E402
from experiment_c.text import parse_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Experiment C reward-hacking outputs.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--monitor-inputs", nargs="+", default=["output_only", "cot_only", "cot_output", "tool_log"])
    parser.add_argument("--baseline-group", default="auto")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--run-name", default=None)
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
    metadata = dict(item.get("metadata", {}) or {})
    manual_keys = [
        "task_success",
        "hack_detected",
        "hack_type",
        "shortcut_used",
        "shortcut_disclosed",
        "behavioral_failure",
        "monitor_detected",
    ]
    manual = dict(metadata.get("manual_audit") or {})
    for key in manual_keys:
        if key in item:
            manual[key] = item[key]
    if manual:
        metadata["manual_audit"] = manual
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
            "metadata": metadata,
        }
    )


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def resolve_baseline(groups: set[str], requested: str) -> str | None:
    if requested != "auto":
        return requested if requested in groups else None
    for candidate in ["Baseline", "C0_outcome_only", "D0_no_mitigation", "A0_base_sft"]:
        if candidate in groups:
            return candidate
    return sorted(groups)[0] if groups else None


def write_manifest(path: Path, args: argparse.Namespace, outputs_count: int, audit_rows: int) -> None:
    input_paths = [args.samples, *args.outputs]
    manifest = {
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "experiment_C/scripts/evaluate_hacking.py",
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(git_value(["status", "--short"])),
        "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in input_paths],
        "outputs_count": outputs_count,
        "audit_rows": audit_rows,
        "monitor_inputs": args.monitor_inputs,
        "bootstrap": {"n_boot": args.n_boot, "seed": args.seed},
        "baseline_group": args.baseline_group,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


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
    paper_summary = paper_table(audits)
    summary_ci = aggregate_with_ci(audits, n_boot=args.n_boot, seed=args.seed)
    baseline_group = resolve_baseline({audit.group for audit in audits}, args.baseline_group)
    comparisons = pairwise_tests(audits, baseline_group, n_boot=args.n_boot, seed=args.seed) if baseline_group else []
    write_jsonl(out_dir / "groundtruth_audit_results.jsonl", rows)
    write_csv(out_dir / "hacking_metrics.csv", summary)
    write_csv(out_dir / "hacking_metrics_with_ci.csv", summary_ci)
    write_csv(out_dir / "pairwise_tests.csv", comparisons)
    write_csv(out_dir / "monitor_input_metrics.csv", monitor_summary)
    write_csv(out_dir / "task_type_metrics.csv", task_summary)
    write_csv(out_dir / "paper_reward_hacking_table.csv", paper_summary)
    (out_dir / "hacking_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(out_dir / "run_manifest.json", args, outputs_count=len(outputs), audit_rows=len(audits))
    print(f"[eval-C] wrote {len(rows)} audit rows")
    print(f"[eval-C] wrote summary to {out_dir / 'hacking_metrics.csv'}")


if __name__ == "__main__":
    main()
