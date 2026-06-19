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

from experiment_a.io import iter_jsonl, write_jsonl  # noqa: E402
from experiment_a.metrics import (  # noqa: E402
    aggregate_metrics,
    aggregate_metrics_with_ci,
    pairwise_tests,
    score_model_output,
    write_csv,
)
from experiment_a.nli import DEFAULT_NLI_MODEL, NLIEntailmentEvaluator, NLIResult, build_answer_hypothesis  # noqa: E402
from experiment_a.schema import CotSample, ModelOutput  # noqa: E402
from experiment_a.text import parse_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Experiment A model outputs.")
    parser.add_argument("--samples-original", required=True)
    parser.add_argument("--samples-hidden", required=True)
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--nli-revision", default=None)
    parser.add_argument("--nli-device", default="auto")
    parser.add_argument("--nli-batch-size", type=int, default=8)
    parser.add_argument("--nli-max-length", type=int, default=512)
    parser.add_argument("--no-nli", action="store_true", help="Disable NLI and use the legacy entailment proxy.")
    parser.add_argument(
        "--allow-nli-fallback",
        action="store_true",
        help="Continue with the entailment proxy if the NLI model cannot be loaded.",
    )
    parser.add_argument("--baseline-group", default="auto")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--run-name", default=None)
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
    metadata = dict(item.get("metadata", {}) or {})
    for key in ["accuracy_judged", "answer_correct", "correct", "is_correct"]:
        if key in item:
            metadata[key] = item[key]
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
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def resolve_baseline_group(groups: set[str], requested: str) -> str | None:
    if requested != "auto":
        return requested if requested in groups else None
    for candidate in ["Baseline", "A0_base_sft", "base_sft"]:
        if candidate in groups:
            return candidate
    return sorted(groups)[0] if groups else None


def build_nli_results(
    outputs: list[ModelOutput],
    samples_by_output: dict[tuple[str, str, str], CotSample],
    args: argparse.Namespace,
) -> dict[tuple[str, str, str], NLIResult]:
    if args.no_nli:
        return {}
    try:
        evaluator = NLIEntailmentEvaluator(
            args.nli_model,
            revision=args.nli_revision,
            device=args.nli_device,
            batch_size=args.nli_batch_size,
            max_length=args.nli_max_length,
        )
    except Exception as exc:
        if args.allow_nli_fallback:
            print(f"[eval][warn] NLI disabled after load failure: {exc}")
            return {}
        raise

    pairs = []
    keys = []
    for out in outputs:
        key = (out.model, out.group, out.prompt_version, out.sample_id)
        sample = samples_by_output[(out.group, out.prompt_version, out.sample_id)]
        pairs.append((out.reasoning, build_answer_hypothesis(sample, out.final_answer)))
        keys.append(key)
    scored = evaluator.score_pairs(pairs)
    return dict(zip(keys, scored))


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    outputs: list[ModelOutput],
    summary_rows: list[dict],
    nli_enabled: bool,
) -> None:
    input_paths = [args.samples_original, args.samples_hidden, *args.outputs]
    manifest = {
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "experiment_A/scripts/evaluate_outputs.py",
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(git_value(["status", "--short"])),
        "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in input_paths],
        "outputs_count": len(outputs),
        "summary_rows": len(summary_rows),
        "nli": {
            "enabled": nli_enabled,
            "model": None if args.no_nli else args.nli_model,
            "revision": args.nli_revision,
            "device": args.nli_device,
            "batch_size": args.nli_batch_size,
            "max_length": args.nli_max_length,
        },
        "bootstrap": {"n_boot": args.n_boot, "seed": args.seed},
        "baseline_group": args.baseline_group,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


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

    samples_by_output: dict[tuple[str, str, str], CotSample] = {}
    for out in outputs:
        sample = hidden_samples[out.sample_id] if out.prompt_version == "hidden_cue" else original_samples[out.sample_id]
        samples_by_output[(out.group, out.prompt_version, out.sample_id)] = sample

    nli_results = build_nli_results(outputs, samples_by_output, args)

    per_sample = []
    for out in outputs:
        sample = samples_by_output[(out.group, out.prompt_version, out.sample_id)]
        original = original_outputs.get((out.model, out.group, out.sample_id))
        nli = nli_results.get((out.model, out.group, out.prompt_version, out.sample_id))
        metric = score_model_output(sample, out, original_output=original, nli_result=nli)
        per_sample.append(metric)

    baseline_group = resolve_baseline_group({metric.group for metric in per_sample}, args.baseline_group)
    per_rows = [metric.to_dict() for metric in per_sample]
    summary = aggregate_metrics(per_sample)
    summary_ci = aggregate_metrics_with_ci(per_sample, n_boot=args.n_boot, seed=args.seed)
    comparisons = (
        pairwise_tests(per_sample, baseline_group=baseline_group, n_boot=args.n_boot, seed=args.seed)
        if baseline_group
        else []
    )

    write_jsonl(out_dir / "per_sample_metrics.jsonl", per_rows)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "summary_with_ci.csv", summary_ci)
    write_csv(out_dir / "pairwise_tests.csv", comparisons)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(out_dir / "run_manifest.json", args, outputs, summary, nli_enabled=bool(nli_results))
    print(f"[eval] wrote {len(per_rows)} per-sample metrics")
    print(f"[eval] wrote summary to {out_dir / 'summary.csv'}")
    print(f"[eval] wrote CI table to {out_dir / 'summary_with_ci.csv'}")
    print(f"[eval] wrote pairwise tests to {out_dir / 'pairwise_tests.csv'}")
    print(f"[eval] wrote run manifest to {out_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
