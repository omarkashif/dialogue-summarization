"""Training-split-only debugging diagnostics for the vanilla Seq2Seq model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from src.evaluate import greedy_decode
from src.models import Seq2Seq
from src.preprocessing import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, Vocabulary
from src.train import train_epoch


@dataclass
class TinySubsetDiagnostic:
    """Losses and periodic greedy-decoding snapshots from a training subset."""

    losses: list[float] = field(default_factory=list)
    snapshots: dict[int, list[dict[str, Any]]] = field(default_factory=dict)


def _content_text(ids: list[int], vocabulary: Vocabulary) -> str:
    tokens = vocabulary.decode(ids)
    return " ".join(
        token for token in tokens if token not in {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN}
    )


@torch.no_grad()
def _greedy_snapshot(
    model: Seq2Seq,
    data_loader: Any,
    vocabulary: Vocabulary,
    device: torch.device,
    max_generation_length: int,
) -> list[dict[str, Any]]:
    """Generate autoregressive predictions for every example in a train-only loader."""
    records: list[dict[str, Any]] = []
    for batch in data_loader:
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
            summary_length = int(batch["summary_lengths"][index].item())
            reference_ids = batch["summary_ids"][index, :summary_length].tolist()
            records.append(
                {
                    "id": str(example_id),
                    "reference_summary": _content_text(reference_ids, vocabulary),
                    "generated_summary": _content_text(generated_ids[index], vocabulary),
                    "generated_eos": generated_eos[index],
                    "hit_max_generation_length": hit_max_length[index],
                }
            )
    return records


def run_tiny_subset_overfit(
    model: Seq2Seq,
    train_subset_loader: Any,
    optimizer: Optimizer,
    criterion: nn.CrossEntropyLoss,
    vocabulary: Vocabulary,
    device: torch.device,
    epochs: int = 200,
    snapshot_every: int = 25,
    max_generation_length: int = 50,
    verbose: bool = False,
) -> TinySubsetDiagnostic:
    """Attempt to memorize a small training-only subset with full teacher forcing.

    This is a debugging test only. It uses no validation data and is not a
    model-selection or reported experiment.
    """
    if epochs < 1 or snapshot_every < 1:
        raise ValueError("epochs and snapshot_every must be at least 1.")

    model.to(device)
    diagnostic = TinySubsetDiagnostic()
    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            model,
            train_subset_loader,
            optimizer,
            criterion,
            device,
            teacher_forcing_ratio=1.0,
        )
        diagnostic.losses.append(loss)
        if epoch == 1 or epoch % snapshot_every == 0 or epoch == epochs:
            diagnostic.snapshots[epoch] = _greedy_snapshot(
                model,
                train_subset_loader,
                vocabulary,
                device,
                max_generation_length,
            )
            if verbose:
                print(f"Diagnostic epoch {epoch:03d} | training loss: {loss:.4f}")
    return diagnostic


def save_tiny_subset_diagnostic(
    diagnostic: TinySubsetDiagnostic, output_path: str | Path
) -> Path:
    """Save debugging losses and snapshots for later inspection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {"losses": diagnostic.losses, "snapshots": diagnostic.snapshots},
            output_file,
            ensure_ascii=False,
            indent=2,
        )
    return output_path
