# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
import inspect
import importlib.metadata
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_a.io import iter_jsonl  # noqa: E402
from experiment_a.prompts import load_prompt_template, render_prompt  # noqa: E402
from experiment_a.rewards import compute_reward, load_reward_config  # noqa: E402
from experiment_a.schema import CotSample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO/RL training for Experiment A.")
    parser.add_argument("--model", required=True, help="A0 SFT checkpoint or base model path.")
    parser.add_argument("--base-model", default=None, help="Optional local base-model path to use when `--model` points to a PEFT adapter checkpoint.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--groups-config", default=str(ROOT / "configs" / "reward_groups.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-template", default=str(ROOT / "prompts" / "qwen_prompt.txt"))
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--use-qlora", action="store_true", help="Load the base model in 4-bit and attach the SFT LoRA adapter.")
    parser.add_argument("--dry-run-rewards", action="store_true", help="Only verify reward function on gold SFT targets.")
    parser.add_argument("--debug-grpo-imports", action="store_true", help="Print GRPO/TRL/vLLM import diagnostics before trainer startup.")
    return parser.parse_args()


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def load_rl_dataset(path: str, template: str):
    from datasets import Dataset

    rows = []
    for item in iter_jsonl(path):
        sample = CotSample.from_dict(item)
        rows.append(
            {
                "prompt": render_prompt(sample, template=template),
                "sample_json": json.dumps(sample.to_dict(), ensure_ascii=False),
                "sample_id": sample.id,
            }
        )
    return Dataset.from_list(rows)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"
    except Exception as exc:
        return f"error: {exc}"


def module_origin(name: str) -> str:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:
        return f"error: {exc}"
    if spec is None:
        return "not-found"
    if spec.origin:
        return str(spec.origin)
    if spec.submodule_search_locations:
        return ", ".join(str(path) for path in spec.submodule_search_locations)
    return "built-in"


def print_grpo_import_diagnostics() -> None:
    print("[debug] python executable:", sys.executable)
    print("[debug] python version:", sys.version.replace("\n", " "))
    for package in ["torch", "transformers", "trl", "vllm", "vllm-ascend", "accelerate", "peft", "bitsandbytes"]:
        print(f"[debug] package {package}: {package_version(package)}")
    for module in ["trl", "trl.extras.vllm_client", "vllm", "vllm_ascend"]:
        print(f"[debug] module {module}: {module_origin(module)}")


def build_grpo_config(GRPOConfig, args: argparse.Namespace):
    signature = inspect.signature(GRPOConfig.__init__)
    supported = set(signature.parameters)
    kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "bf16": args.bf16,
        "logging_steps": 10,
        "save_steps": 200,
        "save_total_limit": 3,
        "report_to": "none",
    }
    if "max_prompt_length" in supported:
        kwargs["max_prompt_length"] = args.max_prompt_length
    else:
        print(
            "[compat] GRPOConfig does not accept max_prompt_length; "
            "prompt truncation will follow the installed TRL version's default behavior."
        )
    return GRPOConfig(**kwargs)


def resolve_model_for_grpo(args: argparse.Namespace):
    model_path = Path(args.model)
    adapter_config = model_path / "adapter_config.json"
    if not adapter_config.exists():
        return args.model

    try:
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except Exception as exc:
        raise SystemExit(f"Missing PEFT model-loading dependencies for adapter checkpoint: {exc}") from exc

    peft_cfg = PeftConfig.from_pretrained(args.model)
    base_model_name = args.base_model or peft_cfg.base_model_name_or_path
    print(f"[compat] Detected PEFT adapter checkpoint at {args.model}")
    print(f"[compat] Loading base model from {base_model_name}")

    quantization_config = None
    model_kwargs = {
        "trust_remote_code": True,
    }
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
        print("[compat] Using 4-bit base-model loading for QLoRA continuation.")
    elif args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    return PeftModel.from_pretrained(base_model, args.model, is_trainable=True)


def main() -> None:
    args = parse_args()
    reward_cfg = load_reward_config(args.groups_config, args.group)
    template = load_prompt_template(args.prompt_template)
    train_ds = load_rl_dataset(args.train_file, template)

    if args.dry_run_rewards:
        from experiment_a.prompts import build_sft_target

        rewards = []
        for row in train_ds:
            sample = CotSample.from_dict(json.loads(row["sample_json"]))
            rewards.append(compute_reward(build_sft_target(sample), sample, reward_cfg))
        avg_reward = sum(rewards) / max(1, len(rewards))
        print(f"[rl] dry-run average reward for {args.group}: {avg_reward:.4f} over {len(rewards)} samples")
        return

    if args.group == "A0_base_sft":
        raise SystemExit("A0_base_sft is SFT-only. Use scripts/train_sft.py instead of RL.")

    try:
        if args.debug_grpo_imports:
            print_grpo_import_diagnostics()
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:
        print("[debug] GRPO import failed; full traceback follows:")
        traceback.print_exc()
        print("[debug] GRPO import diagnostics after failure:")
        print_grpo_import_diagnostics()
        raise SystemExit(
            f"Missing GRPO dependencies or broken import chain. Run `pip install -r requirements.txt`. Details: {exc}"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_ref = resolve_model_for_grpo(args)

    def reward_func(prompts, completions, sample_json=None, **kwargs):
        if sample_json is None:
            sample_json = kwargs.get("sample_json")
        scores = []
        for completion, packed in zip(completions, sample_json):
            sample = CotSample.from_dict(json.loads(packed))
            scores.append(compute_reward(completion_to_text(completion), sample, reward_cfg))
        return scores

    train_args = build_grpo_config(GRPOConfig, args)

    trainer = GRPOTrainer(
        model=model_ref,
        reward_funcs=reward_func,
        args=train_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[rl] saved checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
