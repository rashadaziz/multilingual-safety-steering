from __future__ import annotations

from typing import List

from ._utils import DATA_DIR, read_json_records, sample_examples
from .types import Language, PromptExample

DEFAULT_FILE = DATA_DIR / "json" / "alphasteer" / "malicious_instruct.json"


def load_malicious_instruct(
    num_samples: int | None = None,
    seed: int | None = None,
) -> List[PromptExample]:
    """Load the MaliciousInstruct prompt dataset."""

    dataset_path = DEFAULT_FILE
    examples: List[PromptExample] = []

    for record in read_json_records(dataset_path):
        query = record.get("query")
        if not isinstance(query, str):
            continue
        examples.append(
            PromptExample(
                prompt=query.strip(),
                source="malicious_instruct",
                language=Language.ENGLISH,
            )
        )

    return sample_examples(examples, num_samples, seed)
