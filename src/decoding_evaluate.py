"""Validation-only metric and persistence helpers for decoding comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Callable

import torch
from rouge_score import rouge_scorer

from src.evaluate import ROUGE_METRICS, _content_tokens, _tokens_to_text
from src.preprocessing import Vocabulary


DecodeBatch = Callable[
    [torch.Tensor, torch.Tensor], tuple[list[list[int]], list[bool], list[bool]]
]


@dataclass
class DecodingEvaluation:
    """Validation metrics, decoding time, and one prediction record per example."""

    metrics: dict[str, float]
    predictions: list[dict[str, Any]]


def evaluate_validation_decoding(
    validation_loader: Any,
    vocabulary: Vocabulary,
    device: torch.device,
    decode_batch: DecodeBatch,
) -> DecodingEvaluation:
    """Decode the supplied validation loader and calculate ROUGE/length metrics.

    The caller supplies the decoding function and a validation-only loader. No
    dataset is loaded by this utility, so it has no test-split code path.
    """
    raw_examples = {
        str(example["id"]): example for example in validation_loader.dataset.split
    }
    decoded_batches: list[tuple[dict[str, Any], list[list[int]], list[bool], list[bool]]] = []
    start_time = perf_counter()
    for batch in validation_loader:
        source_ids = batch["dialogue_ids"].to(device)
        source_lengths = batch["dialogue_lengths"].to(device)
        generated_ids, generated_eos, hit_max_length = decode_batch(
            source_ids, source_lengths
        )
        decoded_batches.append((batch, generated_ids, generated_eos, hit_max_length))
    decoding_seconds = perf_counter() - start_time

    scorer = rouge_scorer.RougeScorer(list(ROUGE_METRICS), use_stemmer=True)
    rouge_scores = {metric: [] for metric in ROUGE_METRICS}
    prediction_records: list[dict[str, Any]] = []
    for batch, generated_ids, generated_eos, hit_max_length in decoded_batches:
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
        "decoding_seconds_total": decoding_seconds,
        "decoding_milliseconds_per_example": 1000 * decoding_seconds / len(prediction_records),
    }
    return DecodingEvaluation(metrics=metrics, predictions=prediction_records)


def save_decoding_evaluation(
    evaluation: DecodingEvaluation, output_directory: str | Path
) -> tuple[Path, Path]:
    """Save a new decoding strategy's validation predictions and metrics."""
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
