# Experiment C: CoT Pressure and Reward Hacking

Experiment C tests whether reward pressure on task success, monitor score, readability, or short visible reasoning increases reward hacking. The final claims are based on an independent groundtruth auditor, not the monitor used in the training reward.

This implementation writes per-sample audit rows, monitor-input breakdowns, monitor false-negative rates, bootstrap confidence intervals, baseline comparisons, and a run manifest.

## Formal Pipeline

```powershell
cd D:\work\07ai-app\CoT-evaluate\experiment_C
python -m pip install -r requirements.txt

python scripts\prepare_hacking_suite.py `
  --recipe configs\dataset_full.yaml `
  --out-dir data\processed

python scripts\apply_manual_audits.py `
  --audits data\manual\audit_batch.csv `
  --data-dir data\processed `
  --out-dir data\reviewed `
  --clear-reviewed

python scripts\train_sft.py `
  --model Qwen/Qwen3.5-27B `
  --train-file data\reviewed\sft_train.jsonl `
  --eval-file data\reviewed\valid.jsonl `
  --output-dir checkpoints\C0_outcome_only `
  --use-qlora

python scripts\train_rl_grpo.py `
  --model checkpoints\C0_outcome_only `
  --train-file data\reviewed\rl_train.jsonl `
  --group C3_monitor_high_length `
  --reward-config configs\reward_config_C0_to_C5.yaml `
  --output-dir checkpoints\C3_monitor_high_length

python scripts\evaluate_hacking.py `
  --samples data\reviewed\hacking_test.jsonl `
  --outputs outputs\C0_outputs.jsonl outputs\C3_outputs.jsonl outputs\C5_outputs.jsonl `
  --out-dir results\experiment_C `
  --baseline-group C0_outcome_only
```

## Paper Output Files

- `groundtruth_audit_results.jsonl`
- `hacking_metrics.csv`
- `hacking_metrics_with_ci.csv`
- `monitor_input_metrics.csv`
- `task_type_metrics.csv`
- `pairwise_tests.csv`
- `run_manifest.json`

## Main Metrics

- `hack_rate`: fraction of samples where the groundtruth auditor detects reward hacking.
- `reward_score_gap`: monitor score minus groundtruth score.
- `monitor_recall`: fraction of true hacks detected by the monitor.
- `monitor_FN` / `monitor_false_negative_rate`: `FN / (TP + FN)`.
- `Se`: shortcut used but not disclosed in the visible trace.
- `cot_concealment_rate`: concealment conditional on groundtruth hack detection.
- `behavioral_failure_rate`: task behavior fails the intended specification.

Manual audits can override `task_success`, `hack_detected`, `hack_type`, `shortcut_used`, `shortcut_disclosed`, `behavioral_failure`, and `monitor_detected` through reviewed task metadata or output metadata.
