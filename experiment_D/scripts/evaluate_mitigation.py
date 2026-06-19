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

from experiment_d.io import iter_jsonl, write_jsonl  # noqa: E402
from experiment_d.metrics import aggregate, aggregate_with_ci, pairwise_tests, score_output, stratified, write_csv  # noqa: E402
from experiment_d.rewards import load_mitigation_config  # noqa: E402
from experiment_d.schema import MitigationSample, ModelOutput  # noqa: E402
from experiment_d.text import parse_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Experiment D mitigation outputs.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mitigation-config", default=str(ROOT / "configs" / "mitigation_config_D0_to_D5.yaml"))
    parser.add_argument("--baseline-group", default="auto")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--run-name", default=None)
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
    metadata = dict(item.get("metadata", {}) or {})
    manual = dict(metadata.get("manual_review") or {})
    for key in [
        "accuracy",
        "faithfulness_score",
        "used_hidden_cue",
        "verbalized_hidden_cue",
        "secrecy",
        "over_confession",
        "structured_fields_complete",
        "key_premise_recall",
        "user_satisfaction",
    ]:
        if key in item:
            manual[key] = item[key]
    if manual:
        metadata["manual_review"] = manual
    return ModelOutput.from_dict({"sample_id": item["sample_id"], "model": item["model"], "group": item["group"], "raw_output": item["raw_output"], "reasoning": item["reasoning"], "cue_disclosure": item["cue_disclosure"], "final_answer": item["final_answer"], "output_tokens": item.get("output_tokens"), "input_tokens": item.get("input_tokens"), "latency_s": item.get("latency_s"), "metadata": metadata})


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
    for candidate in ["D0_no_mitigation", "Baseline", "A0_base_sft"]:
        if candidate in groups:
            return candidate
    return sorted(groups)[0] if groups else None


def write_manifest(path: Path, args: argparse.Namespace, outputs_count: int, metrics_count: int) -> None:
    input_paths = [args.samples, args.mitigation_config, *args.outputs]
    manifest = {
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "experiment_D/scripts/evaluate_mitigation.py",
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(git_value(["status", "--short"])),
        "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in input_paths],
        "outputs_count": outputs_count,
        "metrics_count": metrics_count,
        "bootstrap": {"n_boot": args.n_boot, "seed": args.seed},
        "baseline_group": args.baseline_group,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


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
    summary_ci = aggregate_with_ci(metrics, n_boot=args.n_boot, seed=args.seed)
    strata = stratified(metrics)
    baseline_group = resolve_baseline({m.group for m in metrics}, args.baseline_group)
    comparisons = pairwise_tests(metrics, baseline_group, n_boot=args.n_boot, seed=args.seed) if baseline_group else []
    write_jsonl(out_dir / "per_sample_mitigation_metrics.jsonl", rows)
    write_csv(out_dir / "eval_mitigation_metrics.csv", summary)
    write_csv(out_dir / "eval_mitigation_metrics_with_ci.csv", summary_ci)
    write_csv(out_dir / "pairwise_tests.csv", comparisons)
    write_csv(out_dir / "stratified_metrics.csv", strata)
    (out_dir / "eval_mitigation_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(out_dir / "run_manifest.json", args, outputs_count=len(outputs), metrics_count=len(metrics))
    print(f"[eval-D] wrote {len(rows)} per-sample metrics")
    print(f"[eval-D] wrote summary to {out_dir / 'eval_mitigation_metrics.csv'}")


if __name__ == "__main__":
    main()
