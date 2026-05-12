# Experiment A: CoT Readability Pressure

本目录为“实验 A：CoT 可读性压力、忠实度、秘密化程度”提供一套可落地的代码骨架，覆盖：

- 数据集下载与统一 JSONL 格式转换
- hidden cue 干预样本构造
- SFT 训练入口
- GRPO/RL 训练入口与奖励函数
- 推理输出采集
- 指标计算、统计汇总、图表生成
- 人工标注队列生成

## 快速开始

```powershell
cd D:\work\07ai-app\CoT-evaluate\experiment_A
python -m pip install -r requirements.txt

# 1. 准备 pilot 数据。首次运行需要联网下载 Hugging Face 数据集。
python scripts\prepare_dataset.py --recipe configs\dataset_pilot.yaml --out-dir data\processed

# 1.1 可选：抽样给人工标注员，标完后回填到 reviewed 数据目录。
python scripts\make_annotation_sample.py `
  --queue data\processed\annotation_queue.jsonl `
  --out-jsonl data\manual\annotation_batch.jsonl `
  --out-csv data\manual\annotation_batch.csv `
  --n 600

python scripts\apply_manual_annotations.py `
  --annotations data\manual\annotation_batch.csv `
  --data-dir data\processed `
  --out-dir data\reviewed

# 2. 生成 SFT 数据后训练 A0。
python scripts\train_sft.py `
  --model Qwen/Qwen3.5-27B `
  --train-file data\processed\sft_train.jsonl `
  --eval-file data\processed\valid.jsonl `
  --output-dir checkpoints\A0_base_sft `
  --use-qlora

# 3. 从 A0 checkpoint 出发训练 RL 组。
python scripts\train_rl_grpo.py `
  --model checkpoints\A0_base_sft `
  --train-file data\processed\rl_train.jsonl `
  --group A5_mixed `
  --groups-config configs\reward_groups.yaml `
  --output-dir checkpoints\A5_mixed

# 4. 对 test_original / test_hidden_cue 运行推理。
python scripts\run_inference.py `
  --model checkpoints\A5_mixed `
  --samples data\processed\test_original.jsonl `
  --out outputs\qwen_A5_original.jsonl `
  --group A5_mixed

python scripts\run_inference.py `
  --model checkpoints\A5_mixed `
  --samples data\processed\test_hidden_cue.jsonl `
  --out outputs\qwen_A5_hidden_cue.jsonl `
  --group A5_mixed

# 5. 计算指标和生成图表。
python scripts\evaluate_outputs.py `
  --samples-original data\processed\test_original.jsonl `
  --samples-hidden data\processed\test_hidden_cue.jsonl `
  --outputs outputs\qwen_A5_original.jsonl outputs\qwen_A5_hidden_cue.jsonl `
  --out-dir results

python scripts\make_figures.py --summary results\summary.csv --out-dir figures
```

## 数据文件

`prepare_dataset.py` 会生成：

- `sft_train.jsonl`：SFT 监督训练样本
- `rl_train.jsonl`：RL/GRPO 训练样本
- `valid.jsonl`：验证样本
- `test_original.jsonl`：无 hidden cue 的测试版本
- `test_hidden_cue.jsonl`：带 hidden cue 的配对测试版本
- `annotation_queue.jsonl`：需要人工复核或补充的样本

统一样本字段见 `src/experiment_a/schema.py`。核心字段包括：

- `essential_propositions`
- `hidden_cue`
- `cue_target`
- `needs_human_review`
- `human_review_notes`

## 必须人工增加或复核的部分

这些项目不能完全交给自动脚本，否则会影响论文有效性：

1. **essential propositions 人工确认**
   - 自动脚本会从 gold rationale 中拆句生成候选命题。
   - 人工需要确认哪些是真正“必要推理命题”，删除装饰性解释。

2. **hidden cue 干预有效性复核**
   - 自动脚本会为选择题选择一个非 gold 选项作为 `cue_target`。
   - 人工需要检查 cue 是否只构成外部诱导，而不是泄露新信息或改变题意。

3. **BBQ / Sycophancy 的偏置方向确认**
   - BBQ 原始 label 与偏置方向不总是等价。
   - 人工需要确认 `cue_target` 是否对应“错误但有诱导力”的选项。

4. **TruthfulQA / Sycophancy 开放题判分**
   - 开放式答案很难用字符串完全匹配。
   - 需要固定 evaluator 或人工复核 `answer_correct` 与 `faithfulness`。

5. **代码题单元测试安全复核**
   - HumanEval / MBPP 测试代码可能执行任意 Python。
   - 真实实验前应在隔离容器中运行，并人工检查测试片段来源。

6. **抽样人工标注**
   - 每个模型组至少抽 100 条。
   - 标注：正确性、关键步骤覆盖、hidden cue 披露、自相矛盾、事后合理化、可读性 1-5。

`annotation_queue.jsonl` 中已经把这些需要人工介入的原因写入 `needs_human_review` 字段。

## 训练组

奖励组配置在 `configs/reward_groups.yaml`：

- `A0_base_sft`
- `A1_low_length`
- `A2_high_length`
- `A3_fluency`
- `A4_logic_density`
- `A5_mixed`

黑箱组 GPT-5.5 Thinking / Kimi K2 Thinking 不进入训练，只把输出保存成同样的 JSONL 格式后进入评测。

## 重要说明

当前奖励函数中的 fluency、faithfulness、entailment、contradiction 是可运行的启发式版本，适合跑通 pipeline 和做 pilot。正式论文实验建议替换为固定 evaluator 或 NLI 模型，并在报告中声明 evaluator、提示词、版本和抽样人工一致性。
