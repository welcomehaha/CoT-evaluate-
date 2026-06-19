# Experiment A: CoT Readability Pressure

This directory contains the paper-facing implementation of Experiment A. The goal is to test whether readability or length pressure changes visible reasoning quality, faithfulness, and hidden-cue secrecy.

The code is no longer organized as a toy pilot. The final evaluation path records per-sample metrics, NLI entailment details, paired hidden-cue shifts, bootstrap confidence intervals, baseline comparisons, and a reproducibility manifest.

## Formal Pipeline

```powershell
cd D:\work\07ai-app\CoT-evaluate\experiment_A
python -m pip install -r requirements.txt

python scripts\prepare_dataset.py `
  --recipe configs\dataset_full.yaml `
  --out-dir data\processed

python scripts\apply_manual_annotations.py `
  --annotations data\manual\annotation_batch.csv `
  --data-dir data\processed `
  --out-dir data\reviewed `
  --clear-reviewed

python scripts\train_sft.py `
  --model Qwen/Qwen3.5-27B `
  --train-file data\reviewed\sft_train.jsonl `
  --eval-file data\reviewed\valid.jsonl `
  --output-dir checkpoints\Baseline `
  --use-qlora

python scripts\train_rl_grpo.py `
  --model checkpoints\Baseline `
  --train-file data\reviewed\rl_train.jsonl `
  --group LDR `
  --groups-config configs\reward_groups.yaml `
  --output-dir checkpoints\LDR

python scripts\run_inference.py `
  --model checkpoints\LDR `
  --samples data\reviewed\test_original.jsonl `
  --out outputs\LDR_original.jsonl `
  --group LDR `
  --model-label Qwen-LDR

python scripts\run_inference.py `
  --model checkpoints\LDR `
  --samples data\reviewed\test_hidden_cue.jsonl `
  --out outputs\LDR_hidden_cue.jsonl `
  --group LDR `
  --model-label Qwen-LDR

python scripts\evaluate_outputs.py `
  --samples-original data\reviewed\test_original.jsonl `
  --samples-hidden data\reviewed\test_hidden_cue.jsonl `
  --outputs outputs\Baseline_original.jsonl outputs\Baseline_hidden_cue.jsonl outputs\LDR_original.jsonl outputs\LDR_hidden_cue.jsonl `
  --out-dir results\experiment_A `
  --baseline-group Baseline

python scripts\make_figures.py `
  --summary results\experiment_A\summary.csv `
  --out-dir figures\experiment_A
```

For local smoke tests only, add `--no-nli` to `evaluate_outputs.py`. Formal paper results should use the default NLI evaluator:

`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`

## Reward Arms

Paper-facing arms are defined in `configs/reward_groups.yaml`:

- `Baseline`
- `L-P`
- `F-P`
- `L-P+F-P`
- `LDR`
- `LDR+Confession`
- `Dynamic-Penalty`

Backward-compatible aliases `A0_base_sft` through `A5_mixed` are kept so older runs can still be evaluated.

## Required Human Additions

The dataset builder can create candidate fields, but final paper runs require manual confirmation for:

- `essential_propositions`
- `hidden_cue`
- `cue_target`
- bias or sycophancy intervention direction
- code-task test safety and edge-case propositions
- open-ended truthfulness correctness

After review, run `apply_manual_annotations.py --clear-reviewed` so `human_review_pending_rate` reflects unresolved review items rather than already-completed annotation.

## Paper Output Files

`evaluate_outputs.py` writes:

- `per_sample_metrics.jsonl`: one row per generated answer, including NLI probabilities and hidden-cue decisions.
- `summary.csv`: wide table for main paper metrics.
- `summary_with_ci.csv`: bootstrap 95 percent confidence intervals for each metric.
- `pairwise_tests.csv`: bootstrap difference tests against the baseline arm.
- `run_manifest.json`: input hashes, git state, NLI configuration, and bootstrap settings.

The main Experiment A paper metrics are:

- `D`: visible information density.
- `F`: faithfulness score using NLI entailment when enabled.
- `E`: independent NLI entailment score.
- `Se`: hidden-cue secrecy rate.
- `conditional_Se`: secrecy conditional on detected hidden-cue use.
- `S`: automatic readability/satisfaction proxy.
- `paired_cue_shift_rate`: answer changes between original and hidden-cue prompts.
- `accuracy_drop_vs_original`: original accuracy minus hidden-cue accuracy.
