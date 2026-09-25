"""Loading utilities for the SAMSum dialogue-summarization dataset.

The dataset contains ``id``, ``dialogue``, and ``summary`` fields in its
standard ``train``, ``validation``, and ``test`` splits.
"""

from pathlib import Path
import re
from typing import Optional, Union

from datasets import DatasetDict, load_dataset


DATASET_NAME = "knkarthick/samsum"
_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize text for word-level modelling.

    Text is lowercased. Runs of letters, digits, or underscores form word
    tokens, while every non-whitespace punctuation or symbol character is a
    token of its own. For example, ``"Hi, Sam!"`` becomes
    ``["hi", ",", "sam", "!"]``.
    """
    return _TOKEN_PATTERN.findall(text.lower())


def load_samsum(cache_dir: Optional[Union[str, Path]] = None) -> DatasetDict:
    """Load the standard SAMSum splits from the Hugging Face Hub.

    Args:
        cache_dir: Optional directory for Hugging Face's downloaded dataset
            cache. If omitted, the library's default cache location is used.

    Returns:
        A DatasetDict with ``train``, ``validation``, and ``test`` splits.
    """
    return load_dataset(DATASET_NAME, cache_dir=str(cache_dir) if cache_dir else None)
