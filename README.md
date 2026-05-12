# CoT Evaluate: Chain-of-Thought Readability, Faithfulness, and Secrecy

**Authors:** hubo, linzhanying, hujiayu

**Date:** 2026-05-12

---

## Overview

This repository contains the full experimental pipeline for investigating Chain-of-Thought (CoT) readability pressure, faithfulness, hidden-cue gaps, reward hacking, and mitigation strategies in large language models. The project is organized into four experiments (A/B/C/D) that form a progressive research chain:

| Experiment | Core Question | Key Concepts |
|-----------|---------------|-------------|
| **A** | How does readability pressure affect CoT faithfulness and secrecy? | Readability, length, fluency, logic density, faithfulness `F`, secrecy `Se` |
| **B** | Does the model use hidden cues without disclosing them? | Hidden-cue gap, cue following, verbalization recall, rationalization |
| **C** | Does CoT pressure induce reward hacking? | Reward hacking, monitor audit, groundtruth audit, concealment |
| **D** | Are mitigation strategies effective? | Confession training, structured summary, dynamic penalty, utility `U` |

---

## Experiment A: CoT Readability Pressure

**Purpose:** Establish baseline measurements for how different readability pressures (length constraints, fluency rewards, logic-density rewards) affect the faithfulness `F` and secrecy `Se` of model-generated reasoning chains.

**What it does:**
- Downloads and unifies multiple benchmark datasets (GSM8K, HumanEval, MBPP, BBH) into a common JSONL format
- Constructs hidden-cue intervention samples
- Trains SFT baseline (A0) and five RL/GRPO groups (A1-A5) with varying reward configurations
- Collects inference outputs on both original and hidden-cue test sets
- Computes metrics, statistical summaries, and visualizations
- Generates human annotation queues for essential-proposition validation

**Training Groups:**
- `A0_base_sft` — supervised fine-tuning baseline
- `A1_low_length` — penalizes long outputs
- `A2_high_length` — rewards verbose outputs
- `A3_fluency` — rewards fluent reasoning
- `A4_logic_density` — rewards dense logical steps
- `A5_mixed` — balanced combination

**Key Metrics:** correctness, essential-proposition coverage, hidden-cue disclosure, contradiction, rationalization, readability (1-5).

---

## Experiment B: Hidden Cue Gap

**Purpose:** Measure whether model answers are influenced by hidden cues, bias cues, or reward shortcuts, and whether the model actively discloses this influence in its visible reasoning summary.

**What it does:**
- Builds paired original/cue prompts across multiple cue types (authority, majority, position bias, user preference, incorrect explanation, reward shortcut)
- Evaluates cue-following behavior, verbalization recall, and the gap between cue influence and disclosure
- Quantifies rationalization: plausible-looking reasoning that omits the true cue influence
- Reuses the A0-A5 model groups from Experiment A for causal comparison

**Core Metrics:**
- `cue_following_rate` — how often the model follows the cue
- `verbalization_recall` — how often the model mentions the cue
- `hidden_cue_gap` — difference between cue influence and disclosure
- `Se` / `Se_cond` — secrecy under cue-present conditions
- `accuracy_drop` — performance degradation caused by conflict cues
- `rationalization_rate` — plausible reasoning without true cue disclosure

**Cue Relations:**
- `conflict` — cue points to wrong answer (main analysis focus)
- `consistent` — cue points to correct answer (disclosure-only analysis)
- `neutral` — no explicit answer target (reward shortcut / rationalization analysis)

---

## Experiment C: CoT Pressure and Reward Hacking

**Purpose:** Investigate whether readability and outcome pressures in RL training induce reward-hacking behaviors, where models optimize for monitor-visible scores while bypassing true task requirements.

**What it does:**
- Constructs a reward-hacking task suite spanning four risk domains:
  - `code_repair` — hiding test modifications, hard-coding public examples
  - `reasoning` — using hidden cues or position shortcuts without disclosure
  - `summary` — omitting key facts, fabricating citations, keyword stacking
  - `safety` — templated over-refusal or pseudo-safe packaging of dangerous details
- Trains C0-C5 model groups with varying monitor and readability rewards
- Evaluates with both groundtruth audit and independent monitor audit

**Core Principle:** Monitor rewards used during training **cannot** serve as final safety conclusions. Final hack rates must be determined by an independent groundtruth evaluator or external auditor.

**Training Groups:**
- `C0_outcome_only` — task score only
- `C1_outcome_readability` — task + readability
- `C2_outcome_monitor` — task + monitor
- `C3_monitor_high_length` — monitor + length reward
- `C4_monitor_ldr` — monitor + logic-density reward
- `C5_monitor_confession` — monitor + confession reward

**Key Metrics:** task score, monitor score, hack rate, behavioral failure rate, reward-score gap, monitor recall/precision, `Se`, CoT concealment rate.

---

## Experiment D: Mitigation Strategy Effectiveness

**Purpose:** Test whether mitigation strategies — confession training, structured reasoning summaries, and dynamic length penalties — can improve faithfulness `F` and reduce secrecy `Se` while controlling token cost.

**What it does:**
- Constructs a mixed mitigation test suite combining hidden-cue, reasoning, and safety tasks
- Trains D0-D5 groups with incremental mitigation components
- Evaluates trade-offs between accuracy, faithfulness, secrecy, verbalization, and cost
- Performs stratified analysis across task types and risk levels

**Core Question:** Can confession training, structured summary, and dynamic penalty — alone or combined — improve reasoning faithfulness and lower secrecy without excessive token cost?

**Training Groups:**
- `D0_no_mitigation` — baseline
- `D1_confession` — confession reward only
- `D2_structured_summary` — structured reasoning format only
- `D3_dynamic_penalty` — dynamic length penalty only
- `D4_confession_summary` — confession + structured format
- `D5_full_mitigation` — all three combined

**Key Metrics:**
- `accuracy` — task correctness
- `F` — faithfulness (proposition coverage)
- `Se` — secrecy (undisclosed influence)
- `verbalization_recall` — cue/strategy disclosure rate
- `token_cost` — generation length
- `S` — overall safety score
- `over_confession_rate` — false-positive disclosure rate
- `U` — combined utility metric

---

## Repository Structure

```
.
├── experiment_A/      # CoT readability pressure, faithfulness, secrecy baseline
├── experiment_B/      # Hidden-cue gap, cue following, rationalization
├── experiment_C/      # Reward hacking under CoT pressure
├── experiment_D/      # Mitigation strategy effectiveness
└── README.md          # This file
```

Each experiment directory follows a consistent layout:
- `scripts/` — data preparation, training, inference, evaluation, visualization
- `src/experiment_*/` — reusable modules (schema, I/O, metrics, rewards, builders)
- `configs/` — YAML configurations for datasets, training, and reward groups
- `docs/` — protocols for manual annotation, audit, or review
- `data/` — processed datasets, annotation queues, and manual batches
- `examples/toy/` — toy outputs and results for quick pipeline verification

---

## Quick Start (Per Experiment)

Each experiment is self-contained. Typical workflow:

```powershell
cd experiment_X
python -m pip install -r requirements.txt

# 1. Prepare data
python scripts\prepare_*.py --recipe configs\dataset_pilot.yaml --out-dir data\processed

# 2. Train SFT baseline
python scripts\train_sft.py --model Qwen/Qwen3.5-27B ...

# 3. RL training
python scripts\train_rl_grpo.py --model checkpoints\X0_sft ...

# 4. Inference
python scripts\run_inference.py --model checkpoints\X* ...

# 5. Evaluation & figures
python scripts\evaluate_*.py ...
python scripts\make_figures.py ...
```

See individual `experiment_X/README.md` files for detailed commands and group-specific configurations.

---

## Human Annotation / Audit / Review

All four experiments include manual-validation stages:

| Experiment | Manual Stage | Focus |
|-----------|-------------|-------|
| A | Annotation | `essential_propositions`, `answer_correct`, `code_tests_safe` |
| B | Annotation | `cue_target`, `hidden_cue_valid`, `bias_direction`, `rationalization` |
| C | Audit | Code diff, summary factuality, safety over-refusal, cue disclosure |
| D | Review | Cue existence, disclosure validity, over-confession, structured coverage |

Scripts `make_*_sample.py` generate CSV/JSONL batches for annotators. `apply_manual_*.py` merges validated results back into the dataset pipeline.

---

## Notes

- All Python source files include developer annotations (`# Developer: hubo`, `# Date: 2026-05-12`).
- Reward functions (fluency, faithfulness, entailment, contradiction) in the current code are heuristic versions suitable for pilot runs. For final publication experiments, replace them with fixed evaluators or NLI models, and report evaluator version, prompt, and human consistency.
- Black-box models (e.g., GPT-5.5 Thinking, Kimi K2 Thinking) are used as behavioral baselines only; they do not enter training and therefore cannot support causal mechanism claims.
