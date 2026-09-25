"""Model-independent preprocessing and batching for SAMSum Seq2Seq experiments."""

from collections import Counter
from collections.abc import Iterable, Sequence
from functools import partial
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.data import tokenize


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN)

MIN_FREQ = 2
MAX_DIALOGUE_LENGTH = 300
MAX_SUMMARY_LENGTH = 50


class Vocabulary:
    """A deterministic word vocabulary with mappings in both directions."""

    def __init__(self, tokens: Sequence[str]) -> None:
        """Create a vocabulary from tokens, with special tokens first."""
        all_tokens = list(SPECIAL_TOKENS) + [
            token for token in tokens if token not in SPECIAL_TOKENS
        ]
        if len(all_tokens) != len(set(all_tokens)):
            raise ValueError("Vocabulary tokens must be unique.")

        self.stoi = {token: index for index, token in enumerate(all_tokens)}
        self.itos = {index: token for token, index in self.stoi.items()}

    @classmethod
    def build(
        cls, token_sequences: Iterable[Iterable[str]], min_freq: int = MIN_FREQ
    ) -> "Vocabulary":
        """Build a vocabulary from token sequences using a frequency threshold.

        Ordering is by decreasing frequency, then alphabetically, so identical
        training data always produces identical integer IDs.
        """
        if min_freq < 1:
            raise ValueError("min_freq must be at least 1.")

        counts = Counter(token for sequence in token_sequences for token in sequence)
        retained_tokens = sorted(
            (token for token, count in counts.items() if count >= min_freq),
            key=lambda token: (-counts[token], token),
        )
        return cls(retained_tokens)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Convert tokens to IDs, replacing unseen tokens with ``<UNK>``."""
        return [self.stoi.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> list[str]:
        """Convert IDs to tokens, mapping invalid IDs to ``<UNK>``."""
        tokens = [self.itos.get(int(index), UNK_TOKEN) for index in ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in SPECIAL_TOKENS]
        return tokens

    def __len__(self) -> int:
        return len(self.stoi)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    @property
    def sos_id(self) -> int:
        return self.stoi[SOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS_TOKEN]


def build_samsum_vocabulary(train_split: Any, min_freq: int = MIN_FREQ) -> Vocabulary:
    """Build a shared vocabulary from *only* SAMSum training dialogue and summary text."""
    token_sequences = (
        tokenize(text)
        for field in ("dialogue", "summary")
        for text in train_split[field]
    )
    return Vocabulary.build(token_sequences, min_freq=min_freq)


class SAMSumDataset(Dataset[dict[str, Any]]):
    """Tokenize, truncate, and numericalize one already-selected SAMSum split."""

    def __init__(
        self,
        split: Any,
        vocabulary: Vocabulary,
        max_dialogue_length: int = MAX_DIALOGUE_LENGTH,
        max_summary_length: int = MAX_SUMMARY_LENGTH,
    ) -> None:
        if max_dialogue_length < 1 or max_summary_length < 1:
            raise ValueError("Content length limits must be positive.")
        self.split = split
        self.vocabulary = vocabulary
        self.max_dialogue_length = max_dialogue_length
        self.max_summary_length = max_summary_length

    def __len__(self) -> int:
        return len(self.split)

    def _encode_with_boundaries(self, text: str, max_content_length: int) -> torch.Tensor:
        """Truncate content first, then add ``<SOS>`` and ``<EOS>``."""
        content_ids = self.vocabulary.encode(tokenize(text)[:max_content_length])
        sequence_ids = [self.vocabulary.sos_id, *content_ids, self.vocabulary.eos_id]
        return torch.tensor(sequence_ids, dtype=torch.long)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.split[index]
        dialogue_ids = self._encode_with_boundaries(
            example["dialogue"], self.max_dialogue_length
        )
        summary_ids = self._encode_with_boundaries(
            example["summary"], self.max_summary_length
        )
        return {
            "id": example["id"],
            "dialogue_ids": dialogue_ids,
            "summary_ids": summary_ids,
            "dialogue_length": len(dialogue_ids),
            "summary_length": len(summary_ids),
        }


def collate_samsum(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    """Dynamically pad a batch to its longest dialogue and summary sequence."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    return {
        "ids": [example["id"] for example in batch],
        "dialogue_ids": pad_sequence(
            [example["dialogue_ids"] for example in batch],
            batch_first=True,
            padding_value=pad_id,
        ),
        "summary_ids": pad_sequence(
            [example["summary_ids"] for example in batch],
            batch_first=True,
            padding_value=pad_id,
        ),
        "dialogue_lengths": torch.tensor(
            [example["dialogue_length"] for example in batch], dtype=torch.long
        ),
        "summary_lengths": torch.tensor(
            [example["summary_length"] for example in batch], dtype=torch.long
        ),
    }


def create_dataloader(
    dataset: SAMSumDataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader that dynamically pads SAMSum sequences per batch."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_samsum, pad_id=dataset.vocabulary.pad_id),
    )
