"""Validation-only greedy decoding and metric utilities for the vanilla baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
from rouge_score import rouge_scorer

from src.models import Seq2Seq
from src.preprocessing import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, Vocabulary


ROUGE_METRICS = ("rouge1", "rouge2", "rougeL")
DEFAULT_MAX_GENERATION_LENGTH = 50


@dataclass
class ValidationEvaluation:
    """Aggregate validation metrics and one record for every prediction."""

    metrics: dict[str, float]
    predictions: list[dict[str, Any]]


def _tokens_to_text(tokens: list[str]) -> str:
    """Render tokenized text readably while keeping punctuation attached."""
    no_space_before = {".", ",", "!", "?", ":", ";", "%", ")", "]", "}"}
    no_space_after = {"(", "[", "{"}
    text = ""
    previous = ""
    for token in tokens:
        if not text:
            text = token
        elif token in no_space_before or token == "'" or previous in no_space_after or previous == "'":
            text += token
        else:
            text += f" {token}"
        previous = token
    return text


def _content_tokens(ids: list[int], vocabulary: Vocabulary) -> list[str]:
    """Decode IDs and remove sequence/padding markers, retaining ``<UNK>``."""
    tokens = vocabulary.decode(ids)
    return [token for token in tokens if token not in {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN}]


@torch.no_grad()
def greedy_decode(
    model: Seq2Seq,
    source_ids: torch.Tensor,
    source_lengths: torch.Tensor,
    vocabulary: Vocabulary,
    max_generation_length: int = DEFAULT_MAX_GENERATION_LENGTH,
) -> tuple[list[list[int]], list[bool], list[bool]]:
    """Greedily decode a batch without teacher forcing.

    Generation begins with ``<SOS>``. Each argmax prediction becomes the next
    decoder input. A sequence ends when it produces ``<EOS>`` or, if no EOS is
    produced, after ``max_generation_length`` generated content positions.
    """
    if max_generation_length < 1:
        raise ValueError("max_generation_length must be at least 1.")

    model.eval()
    hidden, cell = model.encoder(source_ids, source_lengths)
    batch_size = source_ids.size(0)
    decoder_input = torch.full(
        (batch_size,), vocabulary.sos_id, dtype=torch.long, device=source_ids.device
    )
    generated_ids: list[list[int]] = [[] for _ in range(batch_size)]
    generated_eos = [False] * batch_size

    for _ in range(max_generation_length):
        logits, hidden, cell = model.decoder(decoder_input, hidden, cell)
        next_ids = logits.argmax(dim=1)

        for index, token_id in enumerate(next_ids.tolist()):
            if generated_eos[index]:
                continue
            generated_ids[index].append(token_id)
            if token_id == vocabulary.eos_id:
                generated_eos[index] = True

        if all(generated_eos):
            break
        # Finished examples remain in the batched decoder but no longer record
        # tokens; EOS is fed to them while active examples feed back argmax IDs.
        decoder_input = torch.where(
            torch.tensor(generated_eos, device=source_ids.device),
            torch.full_like(next_ids, vocabulary.eos_id),
            next_ids,
        )

    hit_max_length = [not ended for ended in generated_eos]
    return generated_ids, generated_eos, hit_max_length


@torch.no_grad()
def evaluate_validation(
    model: Seq2Seq,
    validation_loader: Any,
    vocabulary: Vocabulary,
    device: torch.device,
    max_generation_length: int = DEFAULT_MAX_GENERATION_LENGTH,
) -> ValidationEvaluation:
    """Generate and score predictions from the supplied validation loader only.

    The caller must pass a loader constructed from the validation split. This
    function has no test-split argument and never loads a dataset itself.
    """
    model.to(device)
    model.eval()
    scorer = rouge_scorer.RougeScorer(list(ROUGE_METRICS), use_stemmer=True)

    # The supplied loader's dataset is the sole source for displayed raw text.
    raw_examples = {
        str(example["id"]): example for example in validation_loader.dataset.split
    }
    rouge_scores = {metric: [] for metric in ROUGE_METRICS}
    prediction_records: list[dict[str, Any]] = []

    for batch in validation_loader:
        source_ids = batch["dialogue_ids"].to(device)
        source_lengths = batch["dialogue_lengths"].to(device)
        generated_ids, generated_eos, hit_max_length = greedy_decode(
            model,
            source_ids,
            source_lengths,
            vocabulary,
            max_generation_length=max_generation_length,
        )

        for index, example_id in enumerate(batch["ids"]):
            record_id = str(example_id)
            generated_tokens = _content_tokens(generated_ids[index], vocabulary)
            reference_length = int(batch["summary_lengths"][index].item())
            reference_ids = batch["summary_ids"][index, :reference_length].tolist()
            reference_tokens = _content_tokens(reference_ids, vocabulary)
            generated_text = _tokens_to_text(generated_tokens)
            reference_text = _tokens_to_text(reference_tokens)
            scores = scorer.score(reference_text, generated_text)
            for metric in ROUGE_METRICS:
                rouge_scores[metric].append(scores[metric].fmeasure)

            raw_example = raw_examples[record_id]
            prediction_records.append(
                {
                    "id": record_id,
                    "dialogue": raw_example["dialogue"],
                    "reference_summary": raw_example["summary"],
                    "generated_summary": generated_text,
                    "dialogue_length": int(batch["dialogue_lengths"][index].item()) - 2,
                    "reference_length": len(reference_tokens),
                    "generated_length": len(generated_tokens),
                    "generated_eos": generated_eos[index],
                    "hit_max_generation_length": hit_max_length[index],
                }
            )

    if not prediction_records:
        raise ValueError("The validation loader was empty.")

    generated_lengths = [record["generated_length"] for record in prediction_records]
    reference_lengths = [record["reference_length"] for record in prediction_records]
    metrics = {
        "rouge_1": mean(rouge_scores["rouge1"]),
        "rouge_2": mean(rouge_scores["rouge2"]),
        "rouge_l": mean(rouge_scores["rougeL"]),
        "generated_length_mean": mean(generated_lengths),
        "generated_length_median": median(generated_lengths),
        "reference_length_mean": mean(reference_lengths),
        "reference_length_median": median(reference_lengths),
        "generated_eos_percent": 100 * mean(record["generated_eos"] for record in prediction_records),
        "hit_max_generation_length_percent": 100
        * mean(record["hit_max_generation_length"] for record in prediction_records),
    }
    return ValidationEvaluation(metrics=metrics, predictions=prediction_records)


def save_validation_evaluation(
    evaluation: ValidationEvaluation, output_directory: str | Path
) -> tuple[Path, Path]:
    """Save validation predictions as JSONL and aggregate metrics as JSON."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    predictions_path = output_directory / "predictions.jsonl"
    metrics_path = output_directory / "metrics.json"

    with predictions_path.open("w", encoding="utf-8") as predictions_file:
        for record in evaluation.predictions:
            predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(evaluation.metrics, metrics_file, ensure_ascii=False, indent=2)

    return predictions_path, metrics_path
