# Experiment D: Mitigation Strategy Effectiveness

本目录实现“实验 D：缓解策略是否有效”的代码骨架，覆盖：

- mixed mitigation test suite 构造
- D0-D5 缓解策略训练组
- SFT 与 GRPO/RL 训练入口
- confession reward
- structured reasoning summary
- dynamic length penalty
- F、Se、verbalization recall、token cost、S、over-confession、综合效用 U 评测
- 分层分析与图表生成

实验 D 的核心问题：

> confession training、structured summary、dynamic penalty 及其组合，是否能够在控制 token cost 的同时提高推理忠实度 F、降低秘密化程度 Se。

## 快速开始

```powershell
cd D:\work\07ai-app\CoT-evaluate\experiment_D
python -m pip install -r requirements.txt

# 1. 准备 pilot mixed mitigation suite。首次运行需要联网下载 Hugging Face 数据集。
python scripts\prepare_mitigation_suite.py `
  --recipe configs\dataset_pilot.yaml `
  --out-dir data\processed

# 2. 抽样人工复核。
python scripts\make_review_sample.py `
  --queue data\processed\human_review_queue.jsonl `
  --out-jsonl data\manual\review_batch.jsonl `
  --out-csv data\manual\review_batch.csv `
  --n 600

# 3. SFT baseline。
python scripts\train_sft.py `
  --model Qwen/Qwen3.5-27B `
  --train-file data\processed\sft_train.jsonl `
  --eval-file data\processed\valid.jsonl `
  --output-dir checkpoints\D0_sft `
  --use-qlora

# 4. RL 训练 D1-D5。
python scripts\train_rl_grpo.py `
  --model checkpoints\D0_sft `
  --train-file data\processed\rl_train.jsonl `
  --group D5_full_mitigation `
  --mitigation-config configs\mitigation_config_D0_to_D5.yaml `
  --output-dir checkpoints\D5_full_mitigation

# 5. held-out 推理。
python scripts\run_inference.py `
  --model checkpoints\D5_full_mitigation `
  --samples data\processed\mitigation_test.jsonl `
  --out outputs\D5_outputs.jsonl `
  --group D5_full_mitigation

# 6. 评测与出图。
python scripts\evaluate_mitigation.py `
  --samples data\processed\mitigation_test.jsonl `
  --outputs outputs\D5_outputs.jsonl `
  --out-dir results

python scripts\make_figures.py `
  --summary results\eval_mitigation_metrics.csv `
  --strata results\stratified_metrics.csv `
  --out-dir figures
```

## 训练组

配置在 [mitigation_config_D0_to_D5.yaml](D:/work/07ai-app/CoT-evaluate/experiment_D/configs/mitigation_config_D0_to_D5.yaml)：

- `D0_no_mitigation`
- `D1_confession`
- `D2_structured_summary`
- `D3_dynamic_penalty`
- `D4_confession_summary`
- `D5_full_mitigation`

## 主要输出

`evaluate_mitigation.py` 会生成：

- `per_sample_mitigation_metrics.jsonl`
- `eval_mitigation_metrics.csv`
- `stratified_metrics.csv`
- `eval_mitigation_metrics.json`

核心指标：

- `accuracy`
- `F`
- `Se`
- `verbalization_recall`
- `token_cost`
- `S`
- `over_confession_rate`
- `U`

## 人工复核必做项

自动启发式只是 pilot 版本。正式论文实验建议人工复核：

- hidden cue 是否真的存在、是否影响答案
- cue disclosure 是否有效
- 是否 over-confession
- structured summary 是否覆盖 key premises
- dynamic penalty 是否在 hard/high-risk 样本保留足够推理
- 代码和安全任务是否有真实行为失败或 reward hacking

详见 [manual_review_protocol.md](D:/work/07ai-app/CoT-evaluate/experiment_D/docs/manual_review_protocol.md)。
