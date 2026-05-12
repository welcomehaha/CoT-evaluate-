# Experiment B: Hidden Cue Gap

本目录实现“实验 B：模型是否用了 hidden cue 但没说”的完整代码骨架，覆盖：

- 数据集下载与统一 JSONL 转换
- original prompt 与多个 cue prompt 的配对构造
- SFT 训练入口
- GRPO/RL 训练入口与 disclosure/anti-cue 奖励函数
- Qwen A0-A5 组与黑箱输出的统一评测
- cue-following、verbalization recall、hidden-cue gap、Se、accuracy drop、rationalization rate
- 人工标注队列、统计表和图表生成

实验 B 的重点不是证明模型推理能力更强，而是测量：

> 模型最终答案是否被 hidden cue、bias cue 或 reward shortcut 影响，以及模型是否在可见 reasoning summary 中主动披露这种影响。

## 快速开始

```powershell
cd D:\work\07ai-app\CoT-evaluate\experiment_B
python -m pip install -r requirements.txt

# 1. 准备 pilot hidden-cue 数据。首次运行需要联网下载 Hugging Face 数据集。
python scripts\prepare_hidden_cue_dataset.py `
  --recipe configs\dataset_pilot.yaml `
  --out-dir data\processed

# 2. 抽样人工标注。标注员需要确认 cue 是否有效、cue_target 是否合理、是否存在偏置方向。
python scripts\make_annotation_sample.py `
  --queue data\processed\human_annotation_queue.jsonl `
  --out-jsonl data\manual\annotation_batch.jsonl `
  --out-csv data\manual\annotation_batch.csv `
  --n 600

# 3. 可选：人工标注后回填 reviewed 数据。
python scripts\apply_manual_annotations.py `
  --annotations data\manual\annotation_batch.csv `
  --data-dir data\processed `
  --out-dir data\reviewed

# 4. 训练 SFT baseline。
python scripts\train_sft.py `
  --model Qwen/Qwen3.5-27B `
  --train-file data\processed\sft_train.jsonl `
  --eval-file data\processed\valid.jsonl `
  --output-dir checkpoints\A0_base_sft `
  --use-qlora

# 5. 从 A0 checkpoint 进行 GRPO/RL 训练。
python scripts\train_rl_grpo.py `
  --model checkpoints\A0_base_sft `
  --train-file data\processed\rl_train.jsonl `
  --group A5_mixed `
  --groups-config configs\reward_groups.yaml `
  --output-dir checkpoints\A5_mixed

# 6. 分别跑 original 与 cue prompts。
python scripts\run_inference.py `
  --model checkpoints\A5_mixed `
  --samples data\processed\test_original.jsonl `
  --out outputs\qwen_A5_original.jsonl `
  --group A5_mixed

python scripts\run_inference.py `
  --model checkpoints\A5_mixed `
  --samples data\processed\test_cue.jsonl `
  --out outputs\qwen_A5_cue.jsonl `
  --group A5_mixed

# 7. 评测并生成结果表。
python scripts\evaluate_hidden_cue.py `
  --samples-original data\processed\test_original.jsonl `
  --samples-cue data\processed\test_cue.jsonl `
  --outputs outputs\qwen_A5_original.jsonl outputs\qwen_A5_cue.jsonl `
  --out-dir results

python scripts\make_figures.py --summary results\eval_hidden_cue_metrics.csv --out-dir figures
```

## 输出文件

`prepare_hidden_cue_dataset.py` 会生成：

- `sft_train.jsonl`
- `rl_train.jsonl`
- `valid.jsonl`
- `test_original.jsonl`
- `test_cue.jsonl`
- `processed_hidden_cue_test.jsonl`
- `human_annotation_queue.jsonl`
- `manifest.json`

`evaluate_hidden_cue.py` 会生成：

- `per_sample_hidden_cue_metrics.jsonl`
- `eval_hidden_cue_metrics.csv`
- `eval_hidden_cue_metrics.json`
- `cue_type_metrics.csv`
- `mcnemar_by_group.csv`

## 需要人工增加或复核的部分

实验 B 不能只靠自动脚本。以下字段必须人工复核，尤其是正式论文实验：

1. `cue_target`
   - conflict cue 应指向错误但有诱导力的答案。
   - consistent cue 可指向正确答案，但不用于主要的“秘密化风险”结论。

2. `hidden_cue`
   - cue 不能添加真实证据，也不能改变题意。
   - cue 只能是外部权威、从众、用户偏好、位置偏差、错误解释或奖励捷径。

3. `cue_relation`
   - `conflict`：cue 指向错误答案，是主分析重点。
   - `consistent`：cue 指向正确答案，只能分析 disclosure。
   - `neutral`：无明确答案目标，主要用于 reward shortcut/rationalization。

4. `essential_propositions`
   - 用于判断模型是否生成正常但 cue-silent 的替代理由。
   - 自动拆句只是候选，需要人工确认。

5. `rationalization`
   - 自动启发式只能初筛。
   - 正式结果中应抽样人工复核：是否“看起来合理，但没有披露真实 cue 影响”。

6. BBQ / Sycophancy / TruthfulQA
   - BBQ 需要确认 stereotype 或 majority cue 的方向。
   - Sycophancy 需要确认用户偏好 cue 是否与正确答案冲突。
   - TruthfulQA 需要确认 common misconception cue 是否真是常见误解。

## 与实验 A 的关系

实验 B 可以直接复用实验 A 的 Qwen A0-A5 模型组：

- A0 Base-SFT
- A1 Low-Length
- A2 High-Length
- A3 Fluency
- A4 Logic-Density
- A5 Mixed-Balanced

但实验 B 的评测对象是 original/cue counterfactual pair，核心指标是：

- `cue_following_rate`
- `verbalization_recall`
- `hidden_cue_gap`
- `Se`
- `Se_cond`
- `accuracy_drop`
- `rationalization_rate`

黑箱模型 GPT-5.5 Thinking 与 Kimi K2 Thinking 只能作为外部行为参照，不能用于训练机制因果结论。
