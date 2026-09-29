"""Validation-only analysis helpers for the frozen SAMSum experiments."""

from __future__ import annotations

from collections import Counter
from math import ceil
from typing import Any

from rouge_score import rouge_scorer

from src.evaluate import _tokens_to_text


def empirical_tercile_boundaries(lengths: list[int]) -> tuple[int, int]:
    """Return integer 33rd/67th-percentile boundaries by order statistic.

    For ``n`` sorted lengths, the 33rd and 67th percentile boundaries are the
    values at one-indexed ranks ``ceil(n / 3)`` and ``ceil(2n / 3)``. This
    avoids choosing boundaries from validation performance and ensures bins
    have integer token limits.
    """
    if not lengths:
        raise ValueError("lengths must not be empty.")
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    return sorted_lengths[ceil(n / 3) - 1], sorted_lengths[ceil(2 * n / 3) - 1]


def length_group(length: int, short_boundary: int, medium_boundary: int) -> str:
    """Assign a stored model-source length to fixed short/medium/long bins."""
    if length <= short_boundary:
        return "short"
    if length <= medium_boundary:
        return "medium"
    return "long"


def source_lengths_by_id(dataset: Any) -> dict[str, int]:
    """Map IDs to stored model-source lengths, including ``<SOS>/<EOS>``."""
    lengths: dict[str, int] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        lengths[str(item["id"])] = int(item["dialogue_length"])
    return lengths


def paired_greedy_records(
    vanilla_predictions: list[dict[str, Any]],
    attention_predictions: list[dict[str, Any]],
    source_lengths: dict[str, int],
    short_boundary: int,
    medium_boundary: int,
) -> list[dict[str, Any]]:
    """Join same-ID validation predictions and calculate per-example ROUGE-L."""
    vanilla_by_id = {record["id"]: record for record in vanilla_predictions}
    attention_by_id = {record["id"]: record for record in attention_predictions}
    if set(vanilla_by_id) != set(attention_by_id):
        raise ValueError("Vanilla and attention prediction IDs do not match.")
    if set(vanilla_by_id) != set(source_lengths):
        raise ValueError("Prediction and validation source IDs do not match.")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    paired: list[dict[str, Any]] = []
    for example_id in vanilla_by_id:
        vanilla = vanilla_by_id[example_id]
        attention = attention_by_id[example_id]
        reference = vanilla["reference_summary"]
        vanilla_rouge_l = scorer.score(reference, vanilla["generated_summary"])["rougeL"].fmeasure
        attention_rouge_l = scorer.score(reference, attention["generated_summary"])["rougeL"].fmeasure
        stored_length = source_lengths[example_id]
        paired.append(
            {
                "id": example_id,
                "dialogue": vanilla["dialogue"],
                "reference_summary": reference,
                "vanilla_greedy_prediction": vanilla["generated_summary"],
                "attention_greedy_prediction": attention["generated_summary"],
                "source_token_length": stored_length,
                "source_content_token_length": stored_length - 2,
                "length_group": length_group(stored_length, short_boundary, medium_boundary),
                "vanilla_rouge_l": vanilla_rouge_l,
                "attention_rouge_l": attention_rouge_l,
                "rouge_l_difference_attention_minus_vanilla": attention_rouge_l - vanilla_rouge_l,
            }
        )
    return paired


def stratified_rouge_table(paired_records: list[dict[str, Any]]) -> list[dict[str, float | int | str]]:
    """Calculate mean ROUGE scores by preassigned source-length group."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows: list[dict[str, float | int | str]] = []
    for group in ("short", "medium", "long"):
        records = [record for record in paired_records if record["length_group"] == group]
        if not records:
            raise ValueError(f"No records in {group} group.")
        vanilla_scores = [
            scorer.score(record["reference_summary"], record["vanilla_greedy_prediction"])
            for record in records
        ]
        attention_scores = [
            scorer.score(record["reference_summary"], record["attention_greedy_prediction"])
            for record in records
        ]
        rows.append(
            {
                "length_group": group,
                "validation_examples": len(records),
                "mean_source_token_length": sum(record["source_token_length"] for record in records)
                / len(records),
                "vanilla_rouge_1": sum(score["rouge1"].fmeasure for score in vanilla_scores)
                / len(records),
                "vanilla_rouge_2": sum(score["rouge2"].fmeasure for score in vanilla_scores)
                / len(records),
                "vanilla_rouge_l": sum(score["rougeL"].fmeasure for score in vanilla_scores)
                / len(records),
                "attention_rouge_1": sum(score["rouge1"].fmeasure for score in attention_scores)
                / len(records),
                "attention_rouge_2": sum(score["rouge2"].fmeasure for score in attention_scores)
                / len(records),
                "attention_rouge_l": sum(score["rougeL"].fmeasure for score in attention_scores)
                / len(records),
            }
        )
    return rows


def repeated_bigram_count(text: str) -> int:
    """Count repeated token bigram types as a manual-review repetition heuristic."""
    tokens = text.lower().split()
    bigrams = [tuple(tokens[index : index + 2]) for index in range(len(tokens) - 1)]
    return sum(count - 1 for count in Counter(bigrams).values() if count > 1)


def novel_content_word_count(dialogue: str, generated_summary: str) -> int:
    """Return a lightweight candidate-screening count, not a factuality label."""
    source_words = set(token.lower() for token in dialogue.split() if token.isalpha())
    generated_words = [token.lower() for token in generated_summary.split() if token.isalpha()]
    return sum(word not in source_words for word in generated_words)
