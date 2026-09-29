"""One-time final SAMSum test evaluation for the frozen experimental protocol.

This script intentionally contains no training, tuning, or selection logic.
It reconstructs the deterministic vocabulary from the training split only,
then loads the test split exactly for final frozen-checkpoint evaluation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch
from datasets import load_dataset

from src.attention import AttentionDecoder, AttentionEncoder, AttentionSeq2Seq
from src.attention_evaluate import greedy_decode_attention
from src.beam_search import (
    DEFAULT_LENGTH_PENALTY_ALPHA,
    beam_search_decode_attention,
    beam_search_decode_vanilla,
)
from src.decoding_evaluate import evaluate_validation_decoding, save_decoding_evaluation
from src.evaluate import greedy_decode
from src.models import Decoder, Encoder, Seq2Seq
from src.preprocessing import SAMSumDataset, build_samsum_vocabulary, create_dataloader
from src.train import get_device, set_seed
from src.validation_analysis import paired_greedy_records, source_lengths_by_id, stratified_rouge_table


SEED = 6
BATCH_SIZE = 32
MAX_GENERATION_LENGTH = 50
BEAM_WIDTHS = (3, 5)
LENGTH_PENALTY_ALPHA = DEFAULT_LENGTH_PENALTY_ALPHA
# Frozen before test access, from training-only stored source lengths.
SHORT_BOUNDARY = 71
MEDIUM_BOUNDARY = 149

RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "final_test_evaluation"
VANILLA_CHECKPOINT = PROJECT_ROOT / "notebooks" / "checkpoints" / "vanilla_seq2seq_tf1_best.pt"
ATTENTION_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "attention_seq2seq_tf1_best.pt"


def make_vanilla_model(vocabulary_size: int, pad_id: int) -> Seq2Seq:
    return Seq2Seq(
        Encoder(vocabulary_size, pad_id, 256, 256, 0.2),
        Decoder(vocabulary_size, pad_id, 256, 256, 0.2),
    )


def make_attention_model(vocabulary_size: int, pad_id: int) -> AttentionSeq2Seq:
    return AttentionSeq2Seq(
        AttentionEncoder(vocabulary_size, pad_id, 256, 256, 0.2),
        AttentionDecoder(vocabulary_size, pad_id, 256, 256, 256, 0.2),
    )


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Frozen checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def _write_jsonl(records: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _evaluate_model_strategies(
    model_name: str,
    model: torch.nn.Module,
    test_loader: object,
    vocabulary: object,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate the three frozen decoding conditions and write separate files."""
    if model_name == "Vanilla":
        strategies = [("greedy", lambda source, lengths: greedy_decode(model, source, lengths, vocabulary, MAX_GENERATION_LENGTH))]
        strategies.extend(
            [
                (
                    f"beam-{width}",
                    lambda source, lengths, width=width: beam_search_decode_vanilla(
                        model,
                        source,
                        lengths,
                        vocabulary,
                        width,
                        MAX_GENERATION_LENGTH,
                        LENGTH_PENALTY_ALPHA,
                    ),
                )
                for width in BEAM_WIDTHS
            ]
        )
    else:
        strategies = [("greedy", lambda source, lengths: greedy_decode_attention(model, source, lengths, vocabulary, MAX_GENERATION_LENGTH))]
        strategies.extend(
            [
                (
                    f"beam-{width}",
                    lambda source, lengths, width=width: beam_search_decode_attention(
                        model,
                        source,
                        lengths,
                        vocabulary,
                        width,
                        MAX_GENERATION_LENGTH,
                        LENGTH_PENALTY_ALPHA,
                    ),
                )
                for width in BEAM_WIDTHS
            ]
        )

    evaluations: dict[str, object] = {}
    for strategy_name, decoder in strategies:
        evaluation = evaluate_validation_decoding(test_loader, vocabulary, device, decoder)
        directory = RESULTS_DIRECTORY / f"{model_name.lower()}_{strategy_name.replace('-', '_')}"
        save_decoding_evaluation(evaluation, directory)
        evaluations[strategy_name] = evaluation
        metrics = evaluation.metrics
        print(
            f"{model_name} | {strategy_name} | R1={metrics['rouge_1']:.4f} | "
            f"R2={metrics['rouge_2']:.4f} | RL={metrics['rouge_l']:.4f} | "
            f"decode={metrics['decoding_seconds_total']:.2f}s",
            flush=True,
        )
    return evaluations


def _save_test_summary(
    results: dict[str, dict[str, object]],
    test_dataset: SAMSumDataset,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Save aggregate, paired, and frozen-length-stratified test outputs."""
    table_rows: list[dict[str, object]] = []
    for model_name in ("Vanilla", "Attention"):
        for strategy_name in ("greedy", "beam-3", "beam-5"):
            metrics = results[model_name][strategy_name].metrics
            table_rows.append(
                {
                    "model": model_name,
                    "decoding": strategy_name,
                    "rouge_1": metrics["rouge_1"],
                    "rouge_2": metrics["rouge_2"],
                    "rouge_l": metrics["rouge_l"],
                    "generated_length_mean": metrics["generated_length_mean"],
                    "generated_length_median": metrics["generated_length_median"],
                    "reference_length_mean": metrics["reference_length_mean"],
                    "reference_length_median": metrics["reference_length_median"],
                    "eos_percent": metrics["generated_eos_percent"],
                    "hit_max_length_percent": metrics["hit_max_generation_length_percent"],
                    "decoding_seconds_total": metrics["decoding_seconds_total"],
                    "decoding_milliseconds_per_example": metrics["decoding_milliseconds_per_example"],
                }
            )
    with (RESULTS_DIRECTORY / "final_test_metrics.json").open("w", encoding="utf-8") as output:
        json.dump(
            {
                "protocol": {
                    "vocabulary": "deterministically rebuilt from training split only",
                    "max_generation_length": MAX_GENERATION_LENGTH,
                    "beam_widths": list(BEAM_WIDTHS),
                    "beam_scoring": "sum_log_probability / ((5 + L) / 6) ** 0.6",
                    "length_penalty_alpha": LENGTH_PENALTY_ALPHA,
                    "length_L": "excludes <SOS>; includes emitted <EOS>",
                    "test_note": "final test evaluation after protocol freeze; no tuning or selection follows",
                },
                "rows": table_rows,
            },
            output,
            ensure_ascii=False,
            indent=2,
        )
    with (RESULTS_DIRECTORY / "final_test_metrics.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    vanilla_greedy = results["Vanilla"]["greedy"].predictions
    attention_greedy = results["Attention"]["greedy"].predictions
    paired_greedy = paired_greedy_records(
        vanilla_greedy,
        attention_greedy,
        source_lengths_by_id(test_dataset),
        SHORT_BOUNDARY,
        MEDIUM_BOUNDARY,
    )
    _write_jsonl(paired_greedy, RESULTS_DIRECTORY / "paired_greedy_test_predictions.jsonl")
    validation_style_rows = stratified_rouge_table(paired_greedy)
    stratified_rows = [
        {
            **{key: value for key, value in row.items() if key != "validation_examples"},
            "test_examples": row["validation_examples"],
        }
        for row in validation_style_rows
    ]
    stratified_payload = {
        "frozen_boundaries": {
            "short_max_source_tokens": SHORT_BOUNDARY,
            "medium_max_source_tokens": MEDIUM_BOUNDARY,
            "definition": "stored model source length including <SOS> and <EOS>",
            "derived_from": "training-only validation-analysis protocol",
        },
        "rows": stratified_rows,
    }
    with (RESULTS_DIRECTORY / "frozen_length_stratified_test_results.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(stratified_payload, output, ensure_ascii=False, indent=2)
    with (RESULTS_DIRECTORY / "frozen_length_stratified_test_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(stratified_rows[0]))
        writer.writeheader()
        writer.writerows(stratified_rows)

    strategy_maps = {
        (model_name, strategy_name): {
            record["id"]: record
            for record in results[model_name][strategy_name].predictions
        }
        for model_name in ("Vanilla", "Attention")
        for strategy_name in ("greedy", "beam-3", "beam-5")
    }
    same_id_all_decoding: list[dict[str, object]] = []
    for example_id in strategy_maps[("Vanilla", "greedy")]:
        base = strategy_maps[("Vanilla", "greedy")][example_id]
        same_id_all_decoding.append(
            {
                "id": example_id,
                "dialogue": base["dialogue"],
                "reference_summary": base["reference_summary"],
                "source_content_token_length": base["dialogue_length"],
                "vanilla_greedy": base["generated_summary"],
                "vanilla_beam_3": strategy_maps[("Vanilla", "beam-3")][example_id]["generated_summary"],
                "vanilla_beam_5": strategy_maps[("Vanilla", "beam-5")][example_id]["generated_summary"],
                "attention_greedy": strategy_maps[("Attention", "greedy")][example_id]["generated_summary"],
                "attention_beam_3": strategy_maps[("Attention", "beam-3")][example_id]["generated_summary"],
                "attention_beam_5": strategy_maps[("Attention", "beam-5")][example_id]["generated_summary"],
            }
        )
    _write_jsonl(same_id_all_decoding, RESULTS_DIRECTORY / "same_id_all_decoding_test_predictions.jsonl")
    return table_rows, stratified_rows


def main() -> None:
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}", flush=True)
    # This is the explicitly authorized single test access. Vocabulary remains training-only.
    train_split = load_dataset("knkarthick/samsum", split="train")
    test_split = load_dataset("knkarthick/samsum", split="test")
    vocabulary = build_samsum_vocabulary(train_split)
    test_dataset = SAMSumDataset(test_split, vocabulary)
    test_loader = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    vanilla = _load_checkpoint(
        make_vanilla_model(len(vocabulary), vocabulary.pad_id), VANILLA_CHECKPOINT, device
    )
    attention = _load_checkpoint(
        make_attention_model(len(vocabulary), vocabulary.pad_id), ATTENTION_CHECKPOINT, device
    )
    results = {
        "Vanilla": _evaluate_model_strategies("Vanilla", vanilla, test_loader, vocabulary, device),
        "Attention": _evaluate_model_strategies("Attention", attention, test_loader, vocabulary, device),
    }
    table_rows, stratified_rows = _save_test_summary(results, test_dataset)
    print("\nModel | Decoding | R1 | R2 | RL | Mean length | EOS % | Decode time", flush=True)
    for row in table_rows:
        print(
            f"{row['model']} | {row['decoding']} | {row['rouge_1']:.4f} | "
            f"{row['rouge_2']:.4f} | {row['rouge_l']:.4f} | "
            f"{row['generated_length_mean']:.2f} | {row['eos_percent']:.2f} | "
            f"{row['decoding_seconds_total']:.2f}s",
            flush=True,
        )
    print("\nFrozen length groups | n | mean source length | Vanilla R1/R2/RL | Attention R1/R2/RL")
    for row in stratified_rows:
        print(
            f"{row['length_group']} | {row['test_examples']} | {row['mean_source_token_length']:.2f} | "
            f"{row['vanilla_rouge_1']:.4f}/{row['vanilla_rouge_2']:.4f}/{row['vanilla_rouge_l']:.4f} | "
            f"{row['attention_rouge_1']:.4f}/{row['attention_rouge_2']:.4f}/{row['attention_rouge_l']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
