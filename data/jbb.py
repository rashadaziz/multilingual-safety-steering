from __future__ import annotations

from typing import List

from ._utils import DATA_DIR, read_json_records, sample_examples
from .types import Language, PromptExample

DEFAULT_FILE = DATA_DIR / "json" / "alphasteer" / "jbb_harmful.json"


def load_jbb(
    num_samples: int | None = None,
    seed: int | None = None,
) -> List[PromptExample]:
    """Load the Jigsaw bias benchmark harmful prompt dataset."""

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
                source="jbb_harmful",
                language=Language.ENGLISH,
                metadata=metadata,
            )
        )

    return sample_examples(examples, num_samples, seed)
