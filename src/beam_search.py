"""Length-normalized beam-search decoding for the frozen Seq2Seq models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.attention import AttentionSeq2Seq
from src.evaluate import DEFAULT_MAX_GENERATION_LENGTH
from src.models import Seq2Seq
from src.preprocessing import Vocabulary


DEFAULT_LENGTH_PENALTY_ALPHA = 0.6


@dataclass
class _Beam:
    """One partial hypothesis; token IDs exclude the initial ``<SOS>``."""

    token_ids: list[int]
    log_probability: float
    hidden: Tensor
    cell: Tensor
    ended: bool = False


def length_penalty(length: int, alpha: float = DEFAULT_LENGTH_PENALTY_ALPHA) -> float:
    """Return the standard GNMT length penalty ``((5 + length) / 6) ** alpha``."""
    if length < 1:
        raise ValueError("length must be at least 1.")
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    return ((5.0 + length) / 6.0) ** alpha


def _normalized_score(beam: _Beam, alpha: float) -> float:
    """Rank a hypothesis by cumulative log probability divided by its penalty."""
    return beam.log_probability / length_penalty(len(beam.token_ids), alpha)


def _validate_beam_arguments(
    beam_width: int, max_generation_length: int, length_penalty_alpha: float
) -> None:
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1.")
    if max_generation_length < 1:
        raise ValueError("max_generation_length must be at least 1.")
    if length_penalty_alpha < 0:
        raise ValueError("length_penalty_alpha must be non-negative.")


def _prune(candidates: list[_Beam], beam_width: int, alpha: float) -> list[_Beam]:
    return sorted(candidates, key=lambda beam: _normalized_score(beam, alpha), reverse=True)[
        :beam_width
    ]


@torch.no_grad()
def beam_search_decode_vanilla(
    model: Seq2Seq,
    source_ids: Tensor,
    source_lengths: Tensor,
    vocabulary: Vocabulary,
    beam_width: int,
    max_generation_length: int = DEFAULT_MAX_GENERATION_LENGTH,
    length_penalty_alpha: float = DEFAULT_LENGTH_PENALTY_ALPHA,
) -> tuple[list[list[int]], list[bool], list[bool]]:
    """Decode each batch example with log-probability beam search.

    Partial and final hypotheses are ranked with
    ``sum(log p(token_t)) / ((5 + generated_length) / 6) ** alpha``. Length
    includes an emitted ``<EOS>`` when one is generated and excludes ``<SOS>``.
    """
    _validate_beam_arguments(beam_width, max_generation_length, length_penalty_alpha)
    model.eval()
    all_ids: list[list[int]] = []
    all_eos: list[bool] = []
    all_hit_max: list[bool] = []

    for index in range(source_ids.size(0)):
        one_source = source_ids[index : index + 1]
        one_length = source_lengths[index : index + 1]
        hidden, cell = model.encoder(one_source, one_length)
        beams = [_Beam([], 0.0, hidden, cell)]

        for _ in range(max_generation_length):
            ended_beams = [beam for beam in beams if beam.ended]
            active_beams = [beam for beam in beams if not beam.ended]
            candidates: list[_Beam] = list(ended_beams)
            if active_beams:
                decoder_input = torch.tensor(
                    [
                        vocabulary.sos_id if not beam.token_ids else beam.token_ids[-1]
                        for beam in active_beams
                    ],
                    dtype=torch.long,
                    device=source_ids.device,
                )
                hidden_batch = torch.cat([beam.hidden for beam in active_beams], dim=1)
                cell_batch = torch.cat([beam.cell for beam in active_beams], dim=1)
                logits, next_hidden_batch, next_cell_batch = model.decoder(
                    decoder_input, hidden_batch, cell_batch
                )
                for beam_index, beam in enumerate(active_beams):
                    log_probabilities = torch.log_softmax(logits[beam_index], dim=0)
                    values, token_ids = torch.topk(log_probabilities, k=beam_width)
                    next_hidden = next_hidden_batch[:, beam_index : beam_index + 1]
                    next_cell = next_cell_batch[:, beam_index : beam_index + 1]
                    for value, token_id in zip(values.tolist(), token_ids.tolist()):
                        sequence = [*beam.token_ids, token_id]
                        candidates.append(
                            _Beam(
                                sequence,
                                beam.log_probability + value,
                                next_hidden,
                                next_cell,
                                ended=token_id == vocabulary.eos_id,
                            )
                        )
            beams = _prune(candidates, beam_width, length_penalty_alpha)
            if all(beam.ended for beam in beams):
                break

        best = _prune(beams, 1, length_penalty_alpha)[0]
        all_ids.append(best.token_ids)
        all_eos.append(best.ended)
        all_hit_max.append(not best.ended)
    return all_ids, all_eos, all_hit_max


@torch.no_grad()
def beam_search_decode_attention(
    model: AttentionSeq2Seq,
    source_ids: Tensor,
    source_lengths: Tensor,
    vocabulary: Vocabulary,
    beam_width: int,
    max_generation_length: int = DEFAULT_MAX_GENERATION_LENGTH,
    length_penalty_alpha: float = DEFAULT_LENGTH_PENALTY_ALPHA,
) -> tuple[list[list[int]], list[bool], list[bool]]:
    """Beam decode the additive-attention model with the same scoring rule."""
    _validate_beam_arguments(beam_width, max_generation_length, length_penalty_alpha)
    model.eval()
    all_ids: list[list[int]] = []
    all_eos: list[bool] = []
    all_hit_max: list[bool] = []

    for index in range(source_ids.size(0)):
        one_source = source_ids[index : index + 1]
        one_length = source_lengths[index : index + 1]
        encoder_outputs, hidden, cell = model.encoder(one_source, one_length)
        positions = torch.arange(one_source.size(1), device=source_ids.device)
        source_mask = positions.unsqueeze(0) < one_length.unsqueeze(1)
        beams = [_Beam([], 0.0, hidden, cell)]

        for _ in range(max_generation_length):
            ended_beams = [beam for beam in beams if beam.ended]
            active_beams = [beam for beam in beams if not beam.ended]
            candidates: list[_Beam] = list(ended_beams)
            if active_beams:
                decoder_input = torch.tensor(
                    [
                        vocabulary.sos_id if not beam.token_ids else beam.token_ids[-1]
                        for beam in active_beams
                    ],
                    dtype=torch.long,
                    device=source_ids.device,
                )
                hidden_batch = torch.cat([beam.hidden for beam in active_beams], dim=1)
                cell_batch = torch.cat([beam.cell for beam in active_beams], dim=1)
                beam_count = len(active_beams)
                logits, next_hidden_batch, next_cell_batch, _ = model.decoder(
                    decoder_input,
                    hidden_batch,
                    cell_batch,
                    encoder_outputs.expand(beam_count, -1, -1),
                    source_mask.expand(beam_count, -1),
                )
                for beam_index, beam in enumerate(active_beams):
                    log_probabilities = torch.log_softmax(logits[beam_index], dim=0)
                    values, token_ids = torch.topk(log_probabilities, k=beam_width)
                    next_hidden = next_hidden_batch[:, beam_index : beam_index + 1]
                    next_cell = next_cell_batch[:, beam_index : beam_index + 1]
                    for value, token_id in zip(values.tolist(), token_ids.tolist()):
                        sequence = [*beam.token_ids, token_id]
                        candidates.append(
                            _Beam(
                                sequence,
                                beam.log_probability + value,
                                next_hidden,
                                next_cell,
                                ended=token_id == vocabulary.eos_id,
                            )
                        )
            beams = _prune(candidates, beam_width, length_penalty_alpha)
            if all(beam.ended for beam in beams):
                break

        best = _prune(beams, 1, length_penalty_alpha)[0]
        all_ids.append(best.token_ids)
        all_eos.append(best.ended)
        all_hit_max.append(not best.ended)
    return all_ids, all_eos, all_hit_max
