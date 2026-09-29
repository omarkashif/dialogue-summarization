"""Training-split-only tiny-subset diagnostic for the attention model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from src.attention import AttentionSeq2Seq
from src.attention_evaluate import greedy_decode_attention
from src.diagnostics import TinySubsetDiagnostic, _content_text
from src.preprocessing import Vocabulary
from src.train import train_epoch


@torch.no_grad()
def _attention_snapshot(
    model: AttentionSeq2Seq,
    data_loader: Any,
    vocabulary: Vocabulary,
    device: torch.device,
    max_generation_length: int,
) -> list[dict[str, Any]]:
    """Greedily decode every example in the supplied training-only loader."""
    records: list[dict[str, Any]] = []
    for batch in data_loader:
        source_ids = batch["dialogue_ids"].to(device)
        source_lengths = batch["dialogue_lengths"].to(device)
        generated_ids, generated_eos, hit_max_length = greedy_decode_attention(
            model, source_ids, source_lengths, vocabulary, max_generation_length
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


def run_attention_tiny_subset_overfit(
    model: AttentionSeq2Seq,
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
    """Attempt to memorize 32 training examples with full teacher forcing only."""
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
            diagnostic.snapshots[epoch] = _attention_snapshot(
                model, train_subset_loader, vocabulary, device, max_generation_length
            )
            if verbose:
                print(f"Diagnostic epoch {epoch:03d} | training loss: {loss:.4f}")
    return diagnostic
