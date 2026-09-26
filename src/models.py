"""Vanilla LSTM Seq2Seq components for SAMSum summarization."""

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


EMBEDDING_DIM = 256
HIDDEN_DIM = 256
EMBEDDING_DROPOUT = 0.2


class Encoder(nn.Module):
    """A single-layer, unidirectional LSTM encoder."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        embedding_dropout: float = EMBEDDING_DROPOUT,
    ) -> None:
        super().__init__()
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
    ) -> tuple[Tensor, Tensor]:
        """Encode padded source IDs and return final ``(hidden, cell)`` states.

        Args:
            source_ids: Integer IDs with shape ``(batch, source_length)``.
            source_lengths: Unpadded lengths with shape ``(batch,)``. Supplying
                these prevents right-padding from changing final encoder states.
        """
        embedded = self.embedding_dropout(self.embedding(source_ids))
        if source_lengths is None:
            _, (hidden, cell) = self.lstm(embedded)
        else:
            packed = pack_padded_sequence(
                embedded,
                source_lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, cell) = self.lstm(packed)
        return hidden, cell


class Decoder(nn.Module):
    """A single-layer LSTM decoder that predicts one next-token distribution."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        embedding_dropout: float = EMBEDDING_DROPOUT,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, input_ids: Tensor, hidden: Tensor, cell: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Advance one decoding step.

        Args:
            input_ids: Current decoder input IDs with shape ``(batch,)``.
            hidden: Previous hidden state with shape ``(1, batch, hidden_dim)``.
            cell: Previous cell state with shape ``(1, batch, hidden_dim)``.

        Returns:
            Next-token logits of shape ``(batch, vocab_size)`` and the updated
            hidden and cell states.
        """
        embedded = self.embedding_dropout(self.embedding(input_ids)).unsqueeze(1)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        logits = self.output_projection(output.squeeze(1))
        return logits, hidden, cell


class Seq2Seq(nn.Module):
    """A vanilla LSTM encoder-decoder without attention."""

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
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
    ) -> Tensor:
        """Produce logits for each target token after ``<SOS>``.

        ``target_ids`` must include ``<SOS>`` at position zero and ``<EOS>``
        before any right padding. At each step, the decoder receives either the
        gold previous target token (teacher forcing) or its previous argmax
        prediction. Returned logits align with ``target_ids[:, 1:]``.
        """
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be between 0 and 1.")
        if target_ids.ndim != 2 or target_ids.size(1) < 2:
            raise ValueError("target_ids must have shape (batch, target_length >= 2).")

        hidden, cell = self.encoder(source_ids, source_lengths)
        batch_size, target_length = target_ids.shape
        vocab_size = self.decoder.output_projection.out_features
        logits_by_step = torch.empty(
            batch_size,
            target_length - 1,
            vocab_size,
            device=source_ids.device,
        )

        decoder_input = target_ids[:, 0]
        for step in range(target_length - 1):
            step_logits, hidden, cell = self.decoder(decoder_input, hidden, cell)
            logits_by_step[:, step] = step_logits

            if teacher_forcing_ratio == 1.0:
                decoder_input = target_ids[:, step + 1]
            elif teacher_forcing_ratio == 0.0:
                decoder_input = step_logits.argmax(dim=1)
            else:
                use_teacher = torch.rand((), device=source_ids.device) < teacher_forcing_ratio
                decoder_input = (
                    target_ids[:, step + 1] if use_teacher else step_logits.argmax(dim=1)
                )

        return logits_by_step
