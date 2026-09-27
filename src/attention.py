"""LSTM Seq2Seq components with Bahdanau (additive) attention.

This module is deliberately separate from :mod:`src.models`, which contains
the frozen vanilla baseline.  Both models use the same source/target layout:
batch-first IDs with ``<SOS>`` at target position zero, and returned logits
that align with ``target_ids[:, 1:]``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models import EMBEDDING_DIM, EMBEDDING_DROPOUT, HIDDEN_DIM


class BahdanauAttention(nn.Module):
    """Additive attention over all encoder output states."""

    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int) -> None:
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias=False)
        self.decoder_projection = nn.Linear(decoder_hidden_dim, decoder_hidden_dim, bias=False)
        self.energy_projection = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(
        self,
        encoder_outputs: Tensor,
        decoder_hidden: Tensor,
        source_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(context, weights)`` for one decoder time step.

        Args:
            encoder_outputs: All encoder states, ``(batch, source_length, encoder_hidden)``.
            decoder_hidden: Current decoder state, ``(layers, batch, decoder_hidden)``.
            source_mask: Boolean mask of real source positions, ``(batch, source_length)``.
        """
        if source_mask.dtype != torch.bool:
            raise ValueError("source_mask must have boolean dtype.")

        query = self.decoder_projection(decoder_hidden[-1]).unsqueeze(1)
        keys = self.encoder_projection(encoder_outputs)
        scores = self.energy_projection(torch.tanh(keys + query)).squeeze(-1)

        # The packed encoder guarantees each example has at least its <SOS> and
        # <EOS> positions, so every row retains at least one valid score.
        scores = scores.masked_fill(~source_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class AttentionEncoder(nn.Module):
    """Single-layer unidirectional LSTM encoder that returns all time states."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        embedding_dropout: float = EMBEDDING_DROPOUT,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

    def forward(
        self, source_ids: Tensor, source_lengths: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode source IDs into all output states and final ``(hidden, cell)``.

        Returned encoder outputs have shape ``(batch, source_length, hidden)``.
        When lengths are supplied, padding cannot affect the final states and
        unpacking restores the original padded source length for masking.
        """
        embedded = self.embedding_dropout(self.embedding(source_ids))
        if source_lengths is None:
            encoder_outputs, (hidden, cell) = self.lstm(embedded)
        else:
            packed = pack_padded_sequence(
                embedded,
                source_lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_outputs, (hidden, cell) = self.lstm(packed)
            encoder_outputs, _ = pad_packed_sequence(
                packed_outputs,
                batch_first=True,
                total_length=source_ids.size(1),
            )
        return encoder_outputs, hidden, cell


class AttentionDecoder(nn.Module):
    """One-step LSTM decoder using Bahdanau context at every step."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        encoder_hidden_dim: int = HIDDEN_DIM,
        embedding_dropout: float = EMBEDDING_DROPOUT,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.attention = BahdanauAttention(encoder_hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim + encoder_hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.output_projection = nn.Linear(hidden_dim + encoder_hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        hidden: Tensor,
        cell: Tensor,
        encoder_outputs: Tensor,
        source_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Advance one decoder step and return logits plus attention weights."""
        context, attention_weights = self.attention(
            encoder_outputs, hidden, source_mask
        )
        embedded = self.embedding_dropout(self.embedding(input_ids))
        decoder_input = torch.cat((embedded, context), dim=1).unsqueeze(1)
        output, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
        logits = self.output_projection(torch.cat((output.squeeze(1), context), dim=1))
        return logits, hidden, cell, attention_weights


class AttentionSeq2Seq(nn.Module):
    """LSTM encoder-decoder with masked Bahdanau attention."""

    def __init__(self, encoder: AttentionEncoder, decoder: AttentionDecoder) -> None:
        super().__init__()
        if encoder.lstm.hidden_size != decoder.lstm.hidden_size:
            raise ValueError("Encoder and decoder hidden sizes must match.")
        if encoder.lstm.num_layers != decoder.lstm.num_layers:
            raise ValueError("Encoder and decoder layer counts must match.")
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        source_ids: Tensor,
        target_ids: Tensor,
        source_lengths: Optional[Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return next-token logits, optionally with attention weights.

        Logits have shape ``(batch, target_length - 1, vocab_size)`` and align
        with ``target_ids[:, 1:]``.  If requested, attention weights have shape
        ``(batch, target_length - 1, source_length)``.
        """
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be between 0 and 1.")
        if target_ids.ndim != 2 or target_ids.size(1) < 2:
            raise ValueError("target_ids must have shape (batch, target_length >= 2).")

        if source_lengths is not None:
            source_lengths = source_lengths.to(source_ids.device)
        encoder_outputs, hidden, cell = self.encoder(source_ids, source_lengths)
        if source_lengths is None:
            source_mask = source_ids.ne(self.encoder.pad_id)
        else:
            positions = torch.arange(source_ids.size(1), device=source_ids.device)
            source_mask = positions.unsqueeze(0) < source_lengths.unsqueeze(1)

        decoder_input = target_ids[:, 0]
        logits_by_step: list[Tensor] = []
        weights_by_step: list[Tensor] = []
        for step in range(target_ids.size(1) - 1):
            step_logits, hidden, cell, step_weights = self.decoder(
                decoder_input, hidden, cell, encoder_outputs, source_mask
            )
            logits_by_step.append(step_logits)
            weights_by_step.append(step_weights)

            if teacher_forcing_ratio == 1.0:
                decoder_input = target_ids[:, step + 1]
            elif teacher_forcing_ratio == 0.0:
                decoder_input = step_logits.argmax(dim=1)
            else:
                use_teacher = torch.rand((), device=source_ids.device) < teacher_forcing_ratio
                decoder_input = (
                    target_ids[:, step + 1] if use_teacher else step_logits.argmax(dim=1)
                )

        logits = torch.stack(logits_by_step, dim=1)
        if return_attention:
            return logits, torch.stack(weights_by_step, dim=1)
        return logits
