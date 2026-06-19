# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_b.io import iter_jsonl  # noqa: E402
from experiment_b.prompts import build_sft_text, load_prompt_template  # noqa: E402
from experiment_b.schema import HiddenCueSample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT training for Experiment B.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-template", default=str(ROOT / "prompts" / "qwen_hidden_cue_prompt.txt"))
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    return parser.parse_args()


def load_text_dataset(path: str, template: str):
    from datasets import Dataset

    rows = []
    for item in iter_jsonl(path):
        sample = HiddenCueSample.from_dict(item)
        rows.append({"text": build_sft_text(sample, template=template), "sample_id": sample.id})
    return Dataset.from_list(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sft_run_config.json").write_text(
        json.dumps(
            {"created_at_utc": datetime.now(timezone.utc).isoformat(), "script": "experiment_B/scripts/train_sft.py", "args": vars(args)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from trl import SFTTrainer
    except Exception as exc:
        raise SystemExit(f"Missing training dependencies. Run `pip install -r requirements.txt`. Details: {exc}") from exc

    template = load_prompt_template(args.prompt_template)
    train_ds = load_text_dataset(args.train_file, template)
    eval_ds = load_text_dataset(args.eval_file, template) if args.eval_file else None

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    train_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=10,
        save_steps=200,
        eval_steps=200 if eval_ds is not None else None,
        save_total_limit=3,
        report_to="none",
    )
    try:
        train_args = TrainingArguments(**train_kwargs, eval_strategy="steps" if eval_ds is not None else "no")
    except TypeError:
        train_args = TrainingArguments(**train_kwargs, evaluation_strategy="steps" if eval_ds is not None else "no")

    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            peft_config=peft_config,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=peft_config,
        )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[sft-B] saved checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
