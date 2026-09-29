"""Run the attention diagnostic and frozen-protocol validation experiment.

This script deliberately loads SAMSum's train split for the tiny diagnostic,
then only the train and validation splits for full training/evaluation. It has
no test-split code path.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    # Background Windows consoles often default to cp1252, while SAMSum
    # dialogues may contain emoji and other Unicode characters.
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.attention import AttentionDecoder, AttentionEncoder, AttentionSeq2Seq
from src.attention_diagnostics import run_attention_tiny_subset_overfit
from src.attention_evaluate import (
    evaluate_attention_validation,
    save_attention_validation_evaluation,
)
from src.diagnostics import save_tiny_subset_diagnostic
from src.preprocessing import (
    SAMSumDataset,
    build_samsum_vocabulary,
    collate_samsum,
    create_dataloader,
)
from src.train import (
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MAX_GRAD_NORM,
    SEED,
    fit,
    get_device,
    set_seed,
)


BATCH_SIZE = 32
EMBEDDING_DIM = 256
HIDDEN_DIM = 256
EMBEDDING_DROPOUT = 0.2
TRAINING_TEACHER_FORCING_RATIO = 1.0
VALIDATION_TEACHER_FORCING_RATIO = 1.0
MAX_GENERATION_LENGTH = 50
TINY_SUBSET_EPOCHS = 200
TINY_SUBSET_INDICES = list(range(32))
VANILLA_HIGHLIGHTED_VALIDATION_IDS = [
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

CHECKPOINT_PATH = Path("checkpoints/attention_seq2seq_tf1_best.pt")
RESULTS_DIRECTORY = Path("results/attention_seq2seq_tf1_validation")
DIAGNOSTIC_PATH = Path("results/attention_seq2seq_tiny_overfit_diagnostic.json")
HISTORY_PATH = RESULTS_DIRECTORY / "training_history.json"
LOSS_PLOT_PATH = RESULTS_DIRECTORY / "loss_curve.png"


def make_attention_model(vocabulary_size: int, pad_id: int) -> AttentionSeq2Seq:
    """Create the architecture fixed for the controlled comparison."""
    encoder = AttentionEncoder(
        vocabulary_size, pad_id, EMBEDDING_DIM, HIDDEN_DIM, EMBEDDING_DROPOUT
    )
    decoder = AttentionDecoder(
        vocabulary_size,
        pad_id,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        HIDDEN_DIM,
        EMBEDDING_DROPOUT,
    )
    return AttentionSeq2Seq(encoder, decoder)


def main() -> None:
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}", flush=True)

    # Diagnostic: only this explicitly requested training subset is used.
    train_split = load_dataset("knkarthick/samsum", split="train")
    vocabulary = build_samsum_vocabulary(train_split)
    train_dataset = SAMSumDataset(train_split, vocabulary)
    tiny_loader = DataLoader(
        Subset(train_dataset, TINY_SUBSET_INDICES),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_samsum, pad_id=vocabulary.pad_id),
    )
    tiny_model = make_attention_model(len(vocabulary), vocabulary.pad_id).to(device)
    tiny_diagnostic = run_attention_tiny_subset_overfit(
        model=tiny_model,
        train_subset_loader=tiny_loader,
        optimizer=torch.optim.Adam(tiny_model.parameters(), lr=LEARNING_RATE),
        criterion=nn.CrossEntropyLoss(ignore_index=vocabulary.pad_id),
        vocabulary=vocabulary,
        device=device,
        epochs=TINY_SUBSET_EPOCHS,
        snapshot_every=25,
        max_generation_length=MAX_GENERATION_LENGTH,
        verbose=True,
    )
    save_tiny_subset_diagnostic(tiny_diagnostic, DIAGNOSTIC_PATH)
    final_snapshot = tiny_diagnostic.snapshots[TINY_SUBSET_EPOCHS]
    exact_matches = sum(
        record["reference_summary"] == record["generated_summary"]
        for record in final_snapshot
    )
    print(
        f"Tiny diagnostic final loss: {tiny_diagnostic.losses[-1]:.4f} | "
        f"exact greedy matches: {exact_matches}/32",
        flush=True,
    )
    for index, record in enumerate(final_snapshot[:5], start=1):
        print(
            f"Tiny example {index}\nReference: {record['reference_summary']}\n"
            f"Generated: {record['generated_summary']}",
            flush=True,
        )

    # Reset random state so full training uses the frozen seed independently
    # of the diagnostic's random draws.
    set_seed(SEED)
    validation_split = load_dataset("knkarthick/samsum", split="validation")
    train_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = create_dataloader(
        SAMSumDataset(validation_split, vocabulary), batch_size=BATCH_SIZE, shuffle=False
    )
    model = make_attention_model(len(vocabulary), vocabulary.pad_id).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_id)
    history = fit(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
        max_epochs=MAX_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        teacher_forcing_ratio=TRAINING_TEACHER_FORCING_RATIO,
        validation_teacher_forcing_ratio=VALIDATION_TEACHER_FORCING_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        verbose=True,
    )
    with HISTORY_PATH.open("w", encoding="utf-8") as history_file:
        json.dump(
            {
                "train_losses": history.train_losses,
                "validation_losses": history.validation_losses,
                "best_epoch": history.best_epoch,
                "best_validation_loss": history.best_validation_loss,
            },
            history_file,
            indent=2,
        )

    epochs = range(1, len(history.train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.train_losses, marker="o", label="Training loss")
    plt.plot(epochs, history.validation_losses, marker="o", label="Teacher-forced validation loss")
    plt.axvline(history.best_epoch, linestyle="--", label=f"Best epoch ({history.best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Bahdanau Attention Seq2Seq training and validation loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    evaluation_model = make_attention_model(len(vocabulary), vocabulary.pad_id).to(device)
    evaluation_model.load_state_dict(checkpoint["model_state_dict"])
    evaluation = evaluate_attention_validation(
        evaluation_model,
        validation_loader,
        vocabulary,
        device,
        max_generation_length=MAX_GENERATION_LENGTH,
    )
    predictions_path, metrics_path = save_attention_validation_evaluation(
        evaluation, RESULTS_DIRECTORY
    )
    print(
        f"Best checkpoint: epoch {checkpoint['epoch']} | "
        f"teacher-forced validation loss: {checkpoint['validation_loss']:.4f}",
        flush=True,
    )
    for name, value in evaluation.metrics.items():
        print(f"{name}: {value:.4f}", flush=True)
    print(f"Saved predictions: {predictions_path}\nSaved metrics: {metrics_path}", flush=True)

    by_id = {record["id"]: record for record in evaluation.predictions}
    print("Highlighted validation examples (same IDs as vanilla):", flush=True)
    for example_id in VANILLA_HIGHLIGHTED_VALIDATION_IDS:
        record = by_id[example_id]
        print(
            f"ID {example_id}\nDialogue: {record['dialogue']}\n"
            f"Reference: {record['reference_summary']}\n"
            f"Generated: {record['generated_summary']}\n",
            flush=True,
        )


if __name__ == "__main__":
    main()
