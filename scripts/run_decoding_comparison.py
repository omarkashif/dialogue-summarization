"""Validation-only greedy and beam-search comparison for the frozen models."""

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


SEED = 6
BATCH_SIZE = 32
MAX_GENERATION_LENGTH = 50
BEAM_WIDTHS = (3, 5)
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "validation_decoding_comparison"
VANILLA_CHECKPOINT = PROJECT_ROOT / "notebooks" / "checkpoints" / "vanilla_seq2seq_tf1_best.pt"
ATTENTION_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "attention_seq2seq_tf1_best.pt"
HIGHLIGHTED_VALIDATION_IDS = [
    "13817023",
    "13716628",
    "13829420",
    "13819648",
    "13728448",
    "13727903",
    "13821290",
    "13717185",
    "13862374",
    "13817283",
]


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


def _load_model(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def _strategy_rows(
    model_name: str,
    model: torch.nn.Module,
    validation_loader: object,
    vocabulary: object,
    device: torch.device,
) -> list[tuple[str, object]]:
    """Evaluate greedy plus the requested beam widths; save each separately."""
    if model_name == "Vanilla":
        strategies = [("greedy", lambda source, lengths: greedy_decode(model, source, lengths, vocabulary, MAX_GENERATION_LENGTH))]
        strategies.extend(
            [
                (
                    f"beam-{width}",
                    lambda source, lengths, width=width: beam_search_decode_vanilla(
                        model, source, lengths, vocabulary, width, MAX_GENERATION_LENGTH, DEFAULT_LENGTH_PENALTY_ALPHA
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
                        model, source, lengths, vocabulary, width, MAX_GENERATION_LENGTH, DEFAULT_LENGTH_PENALTY_ALPHA
                    ),
                )
                for width in BEAM_WIDTHS
            ]
        )

    rows: list[tuple[str, object]] = []
    for strategy_name, decoder in strategies:
        evaluation = evaluate_validation_decoding(
            validation_loader, vocabulary, device, decoder
        )
        directory_name = f"{model_name.lower()}_{strategy_name.replace('-', '_')}"
        save_decoding_evaluation(evaluation, RESULTS_DIRECTORY / directory_name)
        rows.append((strategy_name, evaluation))
        metrics = evaluation.metrics
        print(
            f"{model_name} | {strategy_name} | R1={metrics['rouge_1']:.4f} | "
            f"R2={metrics['rouge_2']:.4f} | RL={metrics['rouge_l']:.4f} | "
            f"decode={metrics['decoding_seconds_total']:.2f}s",
            flush=True,
        )
    return rows


def main() -> None:
    """Run all requested model/decoder combinations without loading test data."""
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}", flush=True)
    # Deliberately request only these two splits.
    train_split = load_dataset("knkarthick/samsum", split="train")
    validation_split = load_dataset("knkarthick/samsum", split="validation")
    vocabulary = build_samsum_vocabulary(train_split)
    validation_loader = create_dataloader(
        SAMSumDataset(validation_split, vocabulary), batch_size=BATCH_SIZE, shuffle=False
    )

    vanilla = _load_model(
        make_vanilla_model(len(vocabulary), vocabulary.pad_id), VANILLA_CHECKPOINT, device
    )
    attention = _load_model(
        make_attention_model(len(vocabulary), vocabulary.pad_id), ATTENTION_CHECKPOINT, device
    )
    results: dict[str, dict[str, object]] = {}
    for model_name, model in (("Vanilla", vanilla), ("Attention", attention)):
        results[model_name] = {}
        for strategy_name, evaluation in _strategy_rows(
            model_name, model, validation_loader, vocabulary, device
        ):
            results[model_name][strategy_name] = evaluation

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
                    "mean_generated_length": metrics["generated_length_mean"],
                    "median_generated_length": metrics["generated_length_median"],
                    "eos_percent": metrics["generated_eos_percent"],
                    "hit_max_length_percent": metrics["hit_max_generation_length_percent"],
                    "decoding_seconds_total": metrics["decoding_seconds_total"],
                    "decoding_milliseconds_per_example": metrics["decoding_milliseconds_per_example"],
                }
            )
    with (RESULTS_DIRECTORY / "comparison_metrics.json").open("w", encoding="utf-8") as output:
        json.dump(
            {
                "length_normalization": {
                    "formula": "sum_log_probability / ((5 + generated_length) / 6) ** alpha",
                    "alpha": DEFAULT_LENGTH_PENALTY_ALPHA,
                    "generated_length_includes_eos": True,
                    "sos_excluded": True,
                },
                "rows": table_rows,
            },
            output,
            ensure_ascii=False,
            indent=2,
        )
    with (RESULTS_DIRECTORY / "comparison_metrics.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    paired_examples: list[dict[str, object]] = []
    prediction_maps = {
        (model_name, strategy): {
            record["id"]: record
            for record in results[model_name][strategy].predictions
        }
        for model_name in ("Vanilla", "Attention")
        for strategy in ("greedy", "beam-3", "beam-5")
    }
    for example_id in HIGHLIGHTED_VALIDATION_IDS:
        vanilla_greedy = prediction_maps[("Vanilla", "greedy")][example_id]
        paired_examples.append(
            {
                "id": example_id,
                "dialogue": vanilla_greedy["dialogue"],
                "reference_summary": vanilla_greedy["reference_summary"],
                "vanilla_greedy": vanilla_greedy["generated_summary"],
                "vanilla_beam_3": prediction_maps[("Vanilla", "beam-3")][example_id]["generated_summary"],
                "vanilla_beam_5": prediction_maps[("Vanilla", "beam-5")][example_id]["generated_summary"],
                "attention_greedy": prediction_maps[("Attention", "greedy")][example_id]["generated_summary"],
                "attention_beam_3": prediction_maps[("Attention", "beam-3")][example_id]["generated_summary"],
                "attention_beam_5": prediction_maps[("Attention", "beam-5")][example_id]["generated_summary"],
            }
        )
    with (RESULTS_DIRECTORY / "paired_highlighted_examples.json").open("w", encoding="utf-8") as output:
        json.dump(paired_examples, output, ensure_ascii=False, indent=2)

    print("\nModel | Decoding | R1 | R2 | RL | Mean Length | EOS % | Decode Time", flush=True)
    for row in table_rows:
        print(
            f"{row['model']} | {row['decoding']} | {row['rouge_1']:.4f} | "
            f"{row['rouge_2']:.4f} | {row['rouge_l']:.4f} | "
            f"{row['mean_generated_length']:.2f} | {row['eos_percent']:.2f} | "
            f"{row['decoding_seconds_total']:.2f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
