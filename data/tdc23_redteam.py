from __future__ import annotations

from typing import List

from ._utils import DATA_DIR, read_json_records, sample_examples
from .types import Language, PromptExample

DEFAULT_FILE = DATA_DIR / "json" / "alphasteer" / "tdc23_redteam.json"


def load_tdc23_redteam(
    num_samples: int | None = None,
    seed: int | None = None,
) -> List[PromptExample]:
    """Load the TDC 2023 redteam prompt dataset."""

    dataset_path = DEFAULT_FILE
    examples: List[PromptExample] = []

    for record in read_json_records(dataset_path):
        query = record.get("query")
        if not isinstance(query, str):
            continue
        category = record.get("category")
        metadata = {"category": category} if isinstance(category, str) else None
        examples.append(
            PromptExample(
                prompt=query.strip(),
                source="tdc23_redteam",
                language=Language.ENGLISH,
                metadata=metadata,
            )
        )

    return sample_examples(examples, num_samples, seed)
