# Developer: hubo
# Date: 2026-06-19

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .schema import CotSample
from .text import normalize_answer


DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


@dataclass
class NLIResult:
    hypothesis: str
    entailment: float
    neutral: float
    contradiction: float
    model_name: str
    model_revision: str | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


def answer_text_for_hypothesis(sample: CotSample, final_answer: str) -> str:
    answer = str(final_answer or "").strip()
    answer_norm = normalize_answer(answer)
    if sample.choices:
        for key, value in sorted(sample.choices.items()):
            if answer_norm == normalize_answer(key):
                return f"option {key}: {value}"
            if answer_norm == normalize_answer(value):
                return f"option {key}: {value}"
    return answer


def build_answer_hypothesis(sample: CotSample, final_answer: str) -> str:
    answer_text = answer_text_for_hypothesis(sample, final_answer)
    if not answer_text:
        answer_text = "the stated final answer"
    return f"The visible reasoning supports the final answer: {answer_text}."


class NLIEntailmentEvaluator:
    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        *,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
        revision: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "NLI evaluation requires torch and transformers. Install experiment_A/requirements.txt."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.model_revision = revision or getattr(self.model.config, "_commit_hash", None)
        self.entailment_idx, self.neutral_idx, self.contradiction_idx = self._resolve_label_indices()

    def _resolve_label_indices(self) -> tuple[int, int, int]:
        labels = {str(label).lower(): int(idx) for label, idx in self.model.config.label2id.items()}
        for idx, label in self.model.config.id2label.items():
            labels[str(label).lower()] = int(idx)
        entailment = self._find_label(labels, "entail")
        neutral = self._find_label(labels, "neutral")
        contradiction = self._find_label(labels, "contrad")
        missing = [
            name
            for name, value in [
                ("entailment", entailment),
                ("neutral", neutral),
                ("contradiction", contradiction),
            ]
            if value is None
        ]
        if missing:
            raise ValueError(f"NLI model label2id does not expose {missing}: {self.model.config.label2id}")
        return int(entailment), int(neutral), int(contradiction)

    @staticmethod
    def _find_label(labels: dict[str, int], needle: str) -> int | None:
        for label, idx in labels.items():
            if needle in label:
                return idx
        return None

    def score_pairs(self, pairs: Iterable[tuple[str, str]]) -> list[NLIResult]:
        pair_list = list(pairs)
        results: list[NLIResult] = []
        for start in range(0, len(pair_list), self.batch_size):
            batch = pair_list[start : start + self.batch_size]
            premises = [premise or "" for premise, _ in batch]
            hypotheses = [hypothesis or "" for _, hypothesis in batch]
            inputs = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**inputs).logits
                probs = self.torch.softmax(logits, dim=-1).detach().cpu()
            for (_, hypothesis), row in zip(batch, probs):
                results.append(
                    NLIResult(
                        hypothesis=hypothesis,
                        entailment=float(row[self.entailment_idx]),
                        neutral=float(row[self.neutral_idx]),
                        contradiction=float(row[self.contradiction_idx]),
                        model_name=self.model_name,
                        model_revision=self.model_revision,
                    )
                )
        return results
