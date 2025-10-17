from __future__ import annotations

from typing import Iterable, List, Mapping, MutableMapping

from datasets import Dataset, load_dataset

from ._utils import sample_examples
from .types import FLORES_LANGUAGE_CODES, Language, PromptExample


def load_flores_plus(
    language: Language,
    num_samples: int | None = None,
    seed: int | None = None,
) -> List[PromptExample]:
    """Load FLORES++ prompts for a given language across default splits."""

    code = FLORES_LANGUAGE_CODES.get(language)
    if code is None:
        raise ValueError(f"No FLORES++ code registered for language '{language.value}'")

    dataset = load_dataset("openlanguagedata/flores_plus")
    if not isinstance(dataset, MutableMapping):
        raise TypeError("Expected FLORES++ load_dataset call to return a DatasetDict-like object")

    examples: List[PromptExample] = []
    default_splits = ("dev",)
    for split in default_splits:
        if split not in dataset:
            continue
        split_dataset = dataset[split]
        iterable: Iterable[Mapping[str, object]] = (
            split_dataset
            if isinstance(split_dataset, Dataset)
            else list(split_dataset)  # type: ignore[list-item]
        )

        for record in iterable:
            if not isinstance(record, Mapping):
                continue
            iso_639_3 = record.get("iso_639_3")
            iso_15924 = record.get("iso_15924")
            text = record.get("text")

            if not (isinstance(iso_639_3, str) and isinstance(iso_15924, str) and isinstance(text, str)):
                continue

            record_code = f"{iso_639_3.strip()}_{iso_15924.strip()}"
            if record_code != code:
                continue

            metadata = {"split": split, "flores_code": record_code}
            if isinstance(record.get("id"), str):
                metadata["id"] = record["id"]  # type: ignore[index]

            examples.append(
                PromptExample(
                    prompt=text.strip(),
                    source="flores_plus",
                    language=language,
                    metadata=metadata,
                )
            )
    return sample_examples(examples, num_samples, seed)
