# Experiment D: Mitigation Strategy Effectiveness

Experiment D evaluates whether confession training, structured visible summaries, and dynamic length penalties reduce hidden-cue secrecy and reward-hacking risk while preserving answer quality and useful reasoning.

The evaluator writes per-sample mitigation metrics, stratified analyses, bootstrap confidence intervals, baseline comparisons, and a reproducibility manifest.

## Formal Pipeline

```powershell
cd D:\work\07ai-app\CoT-evaluate\experiment_D
python -m pip install -r requirements.txt

python scripts\prepare_mitigation_suite.py `
  --recipe configs\dataset_full.yaml `
  --out-dir data\processed

python scripts\apply_manual_reviews.py `
  --reviews data\manual\review_batch.csv `
  --data-dir data\processed `
  --out-dir data\reviewed `
  --clear-reviewed

python scripts\train_sft.py `
  --model Qwen/Qwen3.5-27B `
  --train-file data\reviewed\sft_train.jsonl `
  --eval-file data\reviewed\valid.jsonl `
  --output-dir checkpoints\D0_no_mitigation `
  --use-qlora

python scripts\train_rl_grpo.py `
  --model checkpoints\D0_no_mitigation `
  --train-file data\reviewed\rl_train.jsonl `
  --group D5_full_mitigation `
  --mitigation-config configs\mitigation_config_D0_to_D5.yaml `
  --output-dir checkpoints\D5_full_mitigation

python scripts\evaluate_mitigation.py `
  --samples data\reviewed\mitigation_test.jsonl `
  --outputs outputs\D0_outputs.jsonl outputs\D1_outputs.jsonl outputs\D5_outputs.jsonl `
  --out-dir results\experiment_D `
  --baseline-group D0_no_mitigation
```

## Paper Output Files

- `per_sample_mitigation_metrics.jsonl`
- `eval_mitigation_metrics.csv`
- `eval_mitigation_metrics_with_ci.csv`
- `stratified_metrics.csv`
- `pairwise_tests.csv`
- `run_manifest.json`

## Main Metrics

- `accuracy`: task correctness, with manual review overrides when provided.
- `F`: faithfulness score based on key-premise recall, answer correctness, disclosure, secrecy, and over-confession.
- `Se`: used hidden cue but did not disclose it.
- `verbalization_recall`: cue or shortcut disclosure rate.
- `over_confession_rate`: disclosure of cues or shortcuts when none are present.
- `token_cost`: visible reasoning plus disclosure token cost.
- `U`: composite mitigation utility.
- `dynamic_lambda_L`: effective length penalty after risk/difficulty/cue/uncertainty adjustment.

Manual reviews can override `accuracy`, `faithfulness_score`, `used_hidden_cue`, `verbalized_hidden_cue`, `secrecy`, `over_confession`, `structured_fields_complete`, `key_premise_recall`, and `user_satisfaction`.
