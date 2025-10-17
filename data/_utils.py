from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TypeVar

DATA_DIR = Path(__file__).resolve().parent

def read_json_records(path: Path) -> Iterator[Mapping[str, Any]]:
    """Yield mapping records from a JSON list file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, Mapping):
                yield entry
    elif isinstance(payload, Mapping):
        yield payload

T = TypeVar("T")

def sample_examples(
    examples: Sequence[T],
    num_samples: int | None,
    seed: int = 42,
) -> list[T]:
    """Return a deterministic subset of examples when requested."""

    items = list(examples)
    if num_samples is None or num_samples >= len(items):
        return items
    rng = random.Random(seed)
    indices = rng.sample(range(len(items)), num_samples)
    return [items[idx] for idx in indices]
