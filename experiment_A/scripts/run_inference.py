from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_a.io import iter_jsonl, write_jsonl  # noqa: E402
from experiment_a.prompts import load_prompt_template, render_prompt  # noqa: E402
from experiment_a.schema import CotSample, ModelOutput  # noqa: E402
from experiment_a.text import approx_token_count, parse_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local model inference on Experiment A samples.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--prompt-template", default=str(ROOT / "prompts" / "qwen_prompt.txt"))
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise SystemExit(f"Missing inference dependencies. Run `pip install -r requirements.txt`. Details: {exc}") from exc

    template = load_prompt_template(args.prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, device_map="auto")
    model.eval()

    outputs = []
    samples_iter = list(iter_jsonl(args.samples))
    if args.limit is not None:
        samples_iter = samples_iter[: args.limit]

    for idx, item in enumerate(samples_iter, start=1):
        sample = CotSample.from_dict(item)
        prompt = render_prompt(sample, template=template)
        start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p
        with torch.no_grad():
            generated = model.generate(**inputs, **gen_kwargs)
        completion_ids = generated[0][inputs["input_ids"].shape[-1] :]
        raw = tokenizer.decode(completion_ids, skip_special_tokens=True)
        reasoning, final_answer = parse_output(raw)
        prompt_version = "hidden_cue" if sample.has_hidden_cue else "original"
        outputs.append(
            ModelOutput(
                sample_id=sample.id,
                model=args.model_label or args.model,
                group=args.group,
                prompt_version=prompt_version,
                raw_output=raw,
                reasoning=reasoning,
                final_answer=final_answer,
                output_tokens=approx_token_count(raw),
                input_tokens=int(inputs["input_ids"].shape[-1]),
                latency_s=time.time() - start,
            ).to_dict()
        )
        if idx % 20 == 0:
            print(f"[infer] {idx}/{len(samples_iter)}")

    write_jsonl(args.out, outputs)
    print(f"[infer] wrote {len(outputs)} outputs to {args.out}")


if __name__ == "__main__":
    main()
