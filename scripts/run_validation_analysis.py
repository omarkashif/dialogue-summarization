"""Complete final validation analyses without loading SAMSum's test split."""

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

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset

from src.attention import AttentionDecoder, AttentionEncoder, AttentionSeq2Seq
from src.attention_evaluate import greedy_decode_attention_with_weights
from src.evaluate import _content_tokens, _tokens_to_text
from src.preprocessing import SAMSumDataset, build_samsum_vocabulary
from src.train import get_device, set_seed
from src.validation_analysis import (
    empirical_tercile_boundaries,
    novel_content_word_count,
    paired_greedy_records,
    repeated_bigram_count,
    source_lengths_by_id,
    stratified_rouge_table,
)


SEED = 6
MAX_GENERATION_LENGTH = 50
ANALYSIS_DIRECTORY = PROJECT_ROOT / "results" / "validation_analysis"
FIGURES_DIRECTORY = ANALYSIS_DIRECTORY / "figures"
ATTENTION_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "attention_seq2seq_tf1_best.pt"
VANILLA_GREEDY_PATH = (
    PROJECT_ROOT
    / "results"
    / "validation_decoding_comparison"
    / "vanilla_greedy"
    / "predictions.jsonl"
)
ATTENTION_GREEDY_PATH = (
    PROJECT_ROOT
    / "results"
    / "validation_decoding_comparison"
    / "attention_greedy"
    / "predictions.jsonl"
)
ATTENTION_DECODING_EXAMPLES_PATH = (
    PROJECT_ROOT
    / "results"
    / "validation_decoding_comparison"
    / "paired_highlighted_examples.json"
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def _write_jsonl(records: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _make_attention_model(vocabulary_size: int, pad_id: int) -> AttentionSeq2Seq:
    return AttentionSeq2Seq(
        AttentionEncoder(vocabulary_size, pad_id, 256, 256, 0.2),
        AttentionDecoder(vocabulary_size, pad_id, 256, 256, 256, 0.2),
    )


def _candidate_record(record: dict[str, object], selection_reason: str) -> dict[str, object]:
    """Keep selection metadata separate from any human factuality judgment."""
    return {**record, "selection_reason": selection_reason}


def _select_qualitative_candidates(paired: list[dict[str, object]]) -> dict[str, object]:
    by_delta_desc = sorted(
        paired, key=lambda record: float(record["rouge_l_difference_attention_minus_vanilla"]), reverse=True
    )
    by_difference_asc = sorted(
        paired, key=lambda record: abs(float(record["rouge_l_difference_attention_minus_vanilla"]))
    )
    both_low = sorted(
        paired, key=lambda record: max(float(record["vanilla_rouge_l"]), float(record["attention_rouge_l"]))
    )
    repetition = sorted(
        paired,
        key=lambda record: repeated_bigram_count(str(record["attention_greedy_prediction"])),
        reverse=True,
    )
    unsupported_candidates = sorted(
        paired,
        key=lambda record: novel_content_word_count(
            str(record["dialogue"]), str(record["attention_greedy_prediction"])
        ),
        reverse=True,
    )
    short = [record for record in paired if record["length_group"] == "short"]
    long = [record for record in paired if record["length_group"] == "long"]
    return {
        "selection_caveat": "These are metric- or heuristic-selected manual-review candidates. They are not automatic factuality or semantic-quality labels.",
        "attention_substantially_higher_rouge_l": [
            _candidate_record(record, "largest attention-minus-vanilla per-example ROUGE-L")
            for record in by_delta_desc[:5]
        ],
        "similar_rouge_l": [
            _candidate_record(record, "smallest absolute per-example ROUGE-L difference")
            for record in by_difference_asc[:5]
        ],
        "both_low_rouge_l": [
            _candidate_record(record, "smallest maximum of the two per-example ROUGE-L scores")
            for record in both_low[:5]
        ],
        "attention_repetition_candidates": [
            _candidate_record(
                record,
                "highest repeated-bigram count in attention output; inspect manually rather than treating as a factuality label",
            )
            for record in repetition[:5]
        ],
        "possible_unsupported_detail_candidates": [
            _candidate_record(
                record,
                "highest source-word novelty heuristic in attention output; requires manual inspection and does not establish hallucination",
            )
            for record in unsupported_candidates[:5]
        ],
        "short_dialogue_examples": [
            _candidate_record(record, "short bin under training-derived tercile boundary")
            for record in sorted(short, key=lambda record: int(record["source_token_length"]))[:3]
        ],
        "long_dialogue_examples": [
            _candidate_record(record, "long bin under training-derived tercile boundary")
            for record in sorted(long, key=lambda record: int(record["source_token_length"]), reverse=True)[:3]
        ],
    }


def _save_rouge_l_plot(rows: list[dict[str, object]]) -> Path:
    labels = [str(row["length_group"]).title() for row in rows]
    x = list(range(len(rows)))
    width = 0.36
    plt.figure(figsize=(7.5, 4.8))
    plt.bar(
        [value - width / 2 for value in x],
        [float(row["vanilla_rouge_l"]) for row in rows],
        width=width,
        label="Vanilla greedy",
    )
    plt.bar(
        [value + width / 2 for value in x],
        [float(row["attention_rouge_l"]) for row in rows],
        width=width,
        label="Attention greedy",
    )
    plt.xticks(x, labels)
    plt.xlabel("Training-derived source-dialogue length group")
    plt.ylabel("Mean ROUGE-L (F-measure)")
    plt.title("Validation ROUGE-L by source-dialogue length")
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    output_path = FIGURES_DIRECTORY / "rouge_l_by_length_group.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _heatmap_selection(paired: list[dict[str, object]]) -> list[dict[str, object]]:
    """Choose readable short/medium examples without using ROUGE to define bins."""
    readable = [
        record
        for record in paired
        if 5 <= len(str(record["attention_greedy_prediction"]).split()) <= 25
        and int(record["source_content_token_length"]) <= 125
    ]
    selected: list[dict[str, object]] = []
    for group in ("short", "medium"):
        group_records = [record for record in readable if record["length_group"] == group]
        if group_records:
            selected.append(sorted(group_records, key=lambda record: int(record["source_token_length"]))[0])
    remaining = [record for record in readable if record["id"] not in {item["id"] for item in selected}]
    if remaining:
        selected.append(sorted(remaining, key=lambda record: int(record["source_token_length"]))[len(remaining) // 2])
    return selected[:3]


def _save_attention_heatmaps(
    model: AttentionSeq2Seq,
    validation_dataset: SAMSumDataset,
    vocabulary: object,
    device: torch.device,
    selected: list[dict[str, object]],
) -> list[dict[str, object]]:
    index_by_id = {
        str(validation_dataset.split[index]["id"]): index for index in range(len(validation_dataset))
    }
    manifests: list[dict[str, object]] = []
    for number, record in enumerate(selected, start=1):
        item = validation_dataset[index_by_id[str(record["id"])]]
        source_ids = item["dialogue_ids"].unsqueeze(0).to(device)
        source_length = torch.tensor([item["dialogue_length"]], device=device)
        generated_ids, _, _, attention_by_example = greedy_decode_attention_with_weights(
            model, source_ids, source_length, vocabulary, MAX_GENERATION_LENGTH
        )
        generated = generated_ids[0]
        weights = attention_by_example[0]
        if generated and generated[-1] == vocabulary.eos_id:
            generated = generated[:-1]
            weights = weights[:-1]
        if not generated or not weights:
            continue
        # Remove sequence markers and all right padding from the displayed axes.
        source_tokens = vocabulary.decode(item["dialogue_ids"][1 : item["dialogue_length"] - 1].tolist())
        # Matplotlib's default font may lack emoji glyphs. Escape only those
        # labels so every displayed source position remains identifiable.
        display_source_tokens = [
            token if token.isascii() else token.encode("ascii", "backslashreplace").decode("ascii")
            for token in source_tokens
        ]
        target_tokens = vocabulary.decode(generated)
        matrix = torch.stack(weights).numpy()[:, 1 : item["dialogue_length"] - 1]
        figure_width = max(9, min(22, 0.22 * len(source_tokens)))
        figure_height = max(3.5, min(10, 0.35 * len(target_tokens)))
        plt.figure(figsize=(figure_width, figure_height))
        image = plt.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
        plt.colorbar(image, label="Attention weight")
        plt.xticks(range(len(display_source_tokens)), display_source_tokens, rotation=90, fontsize=7)
        plt.yticks(range(len(target_tokens)), target_tokens, fontsize=8)
        plt.xlabel("Dialogue source tokens (sequence markers and PAD removed)")
        plt.ylabel("Generated summary tokens (EOS removed)")
        plt.title(f"Bahdanau attention heatmap | validation ID {record['id']}")
        plt.tight_layout()
        filename = f"attention_heatmap_{number}_{record['id']}.png"
        output_path = FIGURES_DIRECTORY / filename
        plt.savefig(output_path, dpi=180)
        plt.close()
        manifests.append(
            {
                "id": record["id"],
                "length_group": record["length_group"],
                "source_content_token_length": record["source_content_token_length"],
                "generated_summary": _tokens_to_text(target_tokens),
                "figure": str(output_path.relative_to(PROJECT_ROOT)),
            }
        )
    return manifests


def _save_final_summary(
    stratified_rows: list[dict[str, object]],
    boundaries: dict[str, int],
) -> Path:
    with (PROJECT_ROOT / "results" / "validation_decoding_comparison" / "comparison_metrics.json").open(
        encoding="utf-8"
    ) as source:
        decoding_comparison = json.load(source)
    architecture_rows = [
        row
        for row in decoding_comparison["rows"]
        if row["decoding"] == "greedy"
    ]
    summary = {
        "architecture_comparison_greedy": {
            "vanilla": {
                "best_teacher_forced_validation_loss": 4.4025,
                "trainable_parameters": 17122465,
                "metrics": next(row for row in architecture_rows if row["model"] == "Vanilla"),
            },
            "attention": {
                "best_teacher_forced_validation_loss": 3.9091375362720977,
                "trainable_parameters": 22865569,
                "metrics": next(row for row in architecture_rows if row["model"] == "Attention"),
            },
        },
        "decoding_comparison": decoding_comparison,
        "training_derived_length_boundaries": boundaries,
        "length_stratified_architecture_comparison": stratified_rows,
        "scope": "Validation only; no test data loaded or evaluated.",
    }
    output_path = ANALYSIS_DIRECTORY / "final_validation_summary.json"
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    markdown_path = ANALYSIS_DIRECTORY / "final_validation_summary.md"
    architecture = summary["architecture_comparison_greedy"]
    lines = [
        "# Final validation summary",
        "",
        "Validation only; the SAMSum test split was not loaded or evaluated.",
        "",
        "## Architecture comparison — greedy decoding",
        "",
        "| Model | Validation loss | Parameters | R1 | R2 | RL | Mean / median length | EOS % | Max-length % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("vanilla", "Vanilla"), ("attention", "Attention")):
        entry = architecture[name]
        metric = entry["metrics"]
        lines.append(
            f"| {label} | {entry['best_teacher_forced_validation_loss']:.4f} | "
            f"{entry['trainable_parameters']:,} | {metric['rouge_1']:.4f} | "
            f"{metric['rouge_2']:.4f} | {metric['rouge_l']:.4f} | "
            f"{metric['mean_generated_length']:.2f} / {metric['median_generated_length']:.0f} | "
            f"{metric['eos_percent']:.2f} | {metric['hit_max_length_percent']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Decoding comparison",
            "",
            "| Model | Decoding | R1 | R2 | RL | Mean length | Decode time |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in decoding_comparison["rows"]:
        lines.append(
            f"| {metric['model']} | {metric['decoding']} | {metric['rouge_1']:.4f} | "
            f"{metric['rouge_2']:.4f} | {metric['rouge_l']:.4f} | "
            f"{metric['mean_generated_length']:.2f} | {metric['decoding_seconds_total']:.2f} s |"
        )
    lines.extend(
        [
            "",
            "## Length-stratified architecture comparison",
            "",
            f"Training-derived stored-source boundaries: short ≤ {boundaries['short_max_source_tokens']}; "
            f"medium ≤ {boundaries['medium_max_source_tokens']}; long > {boundaries['medium_max_source_tokens']}. "
            "Stored source length includes `<SOS>` and `<EOS>`.",
            "",
            "| Group | n | Mean source length | Vanilla RL | Attention RL |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in stratified_rows:
        lines.append(
            f"| {row['length_group']} | {row['validation_examples']} | "
            f"{row['mean_source_token_length']:.2f} | {row['vanilla_rouge_l']:.4f} | "
            f"{row['attention_rouge_l']:.4f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    ANALYSIS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = get_device()
    # Deliberately load only train (vocabulary and fixed bins) and validation.
    train_split = load_dataset("knkarthick/samsum", split="train")
    validation_split = load_dataset("knkarthick/samsum", split="validation")
    vocabulary = build_samsum_vocabulary(train_split)
    train_dataset = SAMSumDataset(train_split, vocabulary)
    validation_dataset = SAMSumDataset(validation_split, vocabulary)
    train_lengths = [int(train_dataset[index]["dialogue_length"]) for index in range(len(train_dataset))]
    short_boundary, medium_boundary = empirical_tercile_boundaries(train_lengths)
    boundaries = {
        "short_max_source_tokens": short_boundary,
        "medium_max_source_tokens": medium_boundary,
        "definition": "stored model-source token length, including <SOS> and <EOS> after truncation",
        "method": "training-only empirical tercile order statistics at ranks ceil(n/3) and ceil(2n/3)",
    }

    vanilla_predictions = _read_jsonl(VANILLA_GREEDY_PATH)
    attention_predictions = _read_jsonl(ATTENTION_GREEDY_PATH)
    paired = paired_greedy_records(
        vanilla_predictions,
        attention_predictions,
        source_lengths_by_id(validation_dataset),
        short_boundary,
        medium_boundary,
    )
    _write_jsonl(paired, ANALYSIS_DIRECTORY / "paired_greedy_validation_examples.jsonl")
    stratified_rows = stratified_rouge_table(paired)
    with (ANALYSIS_DIRECTORY / "length_stratified_results.json").open("w", encoding="utf-8") as output:
        json.dump({"boundaries": boundaries, "rows": stratified_rows}, output, ensure_ascii=False, indent=2)
    with (ANALYSIS_DIRECTORY / "length_stratified_results.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(stratified_rows[0]))
        writer.writeheader()
        writer.writerows(stratified_rows)
    rouge_plot = _save_rouge_l_plot(stratified_rows)

    candidates = _select_qualitative_candidates(paired)
    with (ANALYSIS_DIRECTORY / "qualitative_manual_review_candidates.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(candidates, output, ensure_ascii=False, indent=2)
    with ATTENTION_DECODING_EXAMPLES_PATH.open(encoding="utf-8") as source:
        decoding_examples = json.load(source)
    with (ANALYSIS_DIRECTORY / "attention_decoding_strategy_examples.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(decoding_examples[:5], output, ensure_ascii=False, indent=2)

    checkpoint = torch.load(ATTENTION_CHECKPOINT, map_location=device)
    attention_model = _make_attention_model(len(vocabulary), vocabulary.pad_id).to(device)
    attention_model.load_state_dict(checkpoint["model_state_dict"])
    heatmap_manifest = _save_attention_heatmaps(
        attention_model,
        validation_dataset,
        vocabulary,
        device,
        _heatmap_selection(paired),
    )
    with (FIGURES_DIRECTORY / "attention_heatmap_manifest.json").open("w", encoding="utf-8") as output:
        json.dump(heatmap_manifest, output, ensure_ascii=False, indent=2)
    summary_path = _save_final_summary(stratified_rows, boundaries)

    print(f"Training-derived boundaries: short <= {short_boundary}; medium <= {medium_boundary}; long > {medium_boundary}")
    print("\nLength group | n | mean source length | Vanilla R1/R2/RL | Attention R1/R2/RL")
    for row in stratified_rows:
        print(
            f"{row['length_group']} | {row['validation_examples']} | {row['mean_source_token_length']:.2f} | "
            f"{row['vanilla_rouge_1']:.4f}/{row['vanilla_rouge_2']:.4f}/{row['vanilla_rouge_l']:.4f} | "
            f"{row['attention_rouge_1']:.4f}/{row['attention_rouge_2']:.4f}/{row['attention_rouge_l']:.4f}"
        )
    print(f"\nSaved ROUGE-L plot: {rouge_plot}")
    print(f"Saved attention heatmaps: {len(heatmap_manifest)} in {FIGURES_DIRECTORY}")
    print(f"Saved final validation summary: {summary_path}")


if __name__ == "__main__":
    main()
