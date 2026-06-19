# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_c.io import iter_jsonl  # noqa: E402
from experiment_c.prompts import build_sft_target, load_prompt_template, render_prompt  # noqa: E402
from experiment_c.rewards import compute_reward, load_reward_config  # noqa: E402
from experiment_c.schema import HackingTask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO/RL training for Experiment C.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--reward-config", default=str(ROOT / "configs" / "reward_config_C0_to_C5.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-template", default=str(ROOT / "prompts" / "qwen_reward_hacking_prompt.txt"))
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--dry-run-rewards", action="store_true")
    return parser.parse_args()


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "\n".join(str(item.get("content", item)) if isinstance(item, dict) else str(item) for item in completion)
    return str(completion)


def load_rl_dataset(path: str, template: str):
    from datasets import Dataset

    rows = []
    for item in iter_jsonl(path):
        task = HackingTask.from_dict(item)
        rows.append({"prompt": render_prompt(task, template=template), "task_json": json.dumps(task.to_dict(), ensure_ascii=False), "sample_id": task.id})
    return Dataset.from_list(rows)


def main() -> None:
    args = parse_args()
    reward_cfg = load_reward_config(args.reward_config, args.group)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rl_run_config.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "script": "experiment_C/scripts/train_rl_grpo.py",
                "args": vars(args),
                "reward_config": reward_cfg.__dict__,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    template = load_prompt_template(args.prompt_template)
    train_ds = load_rl_dataset(args.train_file, template)
    if args.dry_run_rewards:
        rewards = []
        for row in train_ds:
            task = HackingTask.from_dict(json.loads(row["task_json"]))
            rewards.append(compute_reward(build_sft_target(task), task, reward_cfg))
        print(f"[rl-C] dry-run average reward for {args.group}: {sum(rewards) / max(1, len(rewards)):.4f}")
        return
    if args.group == "C0_outcome_only":
        print("[rl-C] C0 can be SFT-only; continuing because outcome-only RL was explicitly requested.")
    try:
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:
        raise SystemExit(f"Missing GRPO dependencies. Run `pip install -r requirements.txt`. Details: {exc}") from exc
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def reward_func(prompts, completions, task_json=None, **kwargs):
        task_json = task_json or kwargs.get("task_json")
        scores = []
        for completion, packed in zip(completions, task_json):
            task = HackingTask.from_dict(json.loads(packed))
            scores.append(compute_reward(completion_to_text(completion), task, reward_cfg))
        return scores

    train_args = GRPOConfig(output_dir=args.output_dir, learning_rate=args.learning_rate, num_train_epochs=args.epochs, per_device_train_batch_size=args.per_device_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, num_generations=args.num_generations, max_prompt_length=args.max_prompt_length, max_completion_length=args.max_completion_length, bf16=args.bf16, logging_steps=10, save_steps=200, save_total_limit=3, report_to="none")
    trainer = GRPOTrainer(model=args.model, reward_funcs=reward_func, args=train_args, train_dataset=train_ds, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[rl-C] saved checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
