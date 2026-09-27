"""Training utilities for the vanilla SAMSum LSTM Seq2Seq baseline."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from src.models import Seq2Seq


SEED = 6
LEARNING_RATE = 1e-3
MAX_GRAD_NORM = 1.0
TRAIN_TEACHER_FORCING_RATIO = 0.5
VALIDATION_TEACHER_FORCING_RATIO = 0.0
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3


@dataclass
class TrainingHistory:
    """Loss history and the metadata of the best validation checkpoint."""

    train_losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_validation_loss: float = float("inf")


def set_seed(seed: int = SEED) -> None:
    """Seed relevant random generators for as much repeatability as possible."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Use CUDA when it is available; otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    return (
        batch["dialogue_ids"].to(device),
        batch["summary_ids"].to(device),
        batch["dialogue_lengths"].to(device),
    )


def _loss_for_batch(
    model: Seq2Seq,
    source_ids: Tensor,
    target_ids: Tensor,
    source_lengths: Tensor,
    criterion: nn.CrossEntropyLoss,
    teacher_forcing_ratio: float,
) -> tuple[Tensor, int]:
    """Return next-token cross-entropy and its count of non-padding targets."""
    logits = model(
        source_ids,
        target_ids,
        source_lengths=source_lengths,
        teacher_forcing_ratio=teacher_forcing_ratio,
    )
    # ``target_ids[:, 0]`` is <SOS>, an input to the decoder—not a prediction target.
    next_token_targets = target_ids[:, 1:]
    loss = criterion(logits.reshape(-1, logits.size(-1)), next_token_targets.reshape(-1))
    valid_tokens = int((next_token_targets != criterion.ignore_index).sum().item())
    return loss, valid_tokens


def train_epoch(
    model: Seq2Seq,
    data_loader: Any,
    optimizer: Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    teacher_forcing_ratio: float = TRAIN_TEACHER_FORCING_RATIO,
    max_grad_norm: float = MAX_GRAD_NORM,
) -> float:
    """Run one training epoch and return mean cross-entropy per non-pad token."""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in data_loader:
        source_ids, target_ids, source_lengths = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, valid_tokens = _loss_for_batch(
            model,
            source_ids,
            target_ids,
            source_lengths,
            criterion,
            teacher_forcing_ratio,
        )
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    if total_tokens == 0:
        raise ValueError("The training loader contained no non-padding target tokens.")
    return total_loss / total_tokens


def validate_epoch(
    model: Seq2Seq,
    data_loader: Any,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    teacher_forcing_ratio: float = VALIDATION_TEACHER_FORCING_RATIO,
) -> float:
    """Run one validation epoch and return mean cross-entropy per non-pad token."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            source_ids, target_ids, source_lengths = _batch_to_device(batch, device)
            loss, valid_tokens = _loss_for_batch(
                model,
                source_ids,
                target_ids,
                source_lengths,
                criterion,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    if total_tokens == 0:
        raise ValueError("The validation loader contained no non-padding target tokens.")
    return total_loss / total_tokens


def training_sanity_check(
    model: Seq2Seq,
    batch: dict[str, Any],
    optimizer: Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    teacher_forcing_ratio: float = TRAIN_TEACHER_FORCING_RATIO,
    max_grad_norm: float = MAX_GRAD_NORM,
) -> tuple[float, float]:
    """Run one complete optimization step and return ``(loss, preclip_norm)``.

    This intentionally updates the provided model and optimizer. Use a
    disposable model for the smoke test before starting a full experiment.
    """
    model.train()
    source_ids, target_ids, source_lengths = _batch_to_device(batch, device)
    optimizer.zero_grad(set_to_none=True)
    loss, _ = _loss_for_batch(
        model,
        source_ids,
        target_ids,
        source_lengths,
        criterion,
        teacher_forcing_ratio,
    )
    loss.backward()
    preclip_norm = float(clip_grad_norm_(model.parameters(), max_norm=max_grad_norm))
    optimizer.step()
    return loss.item(), preclip_norm


def fit(
    model: Seq2Seq,
    train_loader: Any,
    validation_loader: Any,
    optimizer: Optimizer,
    criterion: nn.CrossEntropyLoss,
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    teacher_forcing_ratio: float = TRAIN_TEACHER_FORCING_RATIO,
    validation_teacher_forcing_ratio: float = VALIDATION_TEACHER_FORCING_RATIO,
    max_grad_norm: float = MAX_GRAD_NORM,
    verbose: bool = True,
) -> TrainingHistory:
    """Train with validation-loss checkpointing and early stopping."""
    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1.")
    if patience < 1:
        raise ValueError("patience must be at least 1.")

    device = get_device() if device is None else device
    model.to(device)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = TrainingHistory()
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            teacher_forcing_ratio=teacher_forcing_ratio,
            max_grad_norm=max_grad_norm,
        )
        validation_loss = validate_epoch(
            model,
            validation_loader,
            criterion,
            device,
            teacher_forcing_ratio=validation_teacher_forcing_ratio,
        )
        history.train_losses.append(train_loss)
        history.validation_losses.append(validation_loss)

        if validation_loss < history.best_validation_loss:
            history.best_validation_loss = validation_loss
            history.best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"Epoch {epoch:02d} | train loss: {train_loss:.4f} | "
                f"validation loss: {validation_loss:.4f}"
            )
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping after {epoch} epochs (patience={patience}).")
            break

    return history
