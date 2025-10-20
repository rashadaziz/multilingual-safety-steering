from typing import Dict, Iterable, List, Mapping, MutableMapping

from datasets import Dataset, load_dataset

from ._utils import sample_examples
from .types import FLORES_LANGUAGE_CODES, Language, PromptExample

dataset = load_dataset("openlanguagedata/flores_plus")
CODE_TO_LANGUAGE: Dict[str, Language] = {code: lang for lang, code in FLORES_LANGUAGE_CODES.items()}
DEFAULT_SPLITS = ("dev",)
_LANGUAGE_PROMPT_CACHE: Dict[str, List[PromptExample]] | None = None


def _ensure_language_cache() -> Dict[str, List[PromptExample]]:
    global _LANGUAGE_PROMPT_CACHE
    if _LANGUAGE_PROMPT_CACHE is not None:
        return _LANGUAGE_PROMPT_CACHE

    if not isinstance(dataset, MutableMapping):
        raise TypeError("Expected FLORES++ load_dataset call to return a DatasetDict-like object")

    cache: Dict[str, List[PromptExample]] = {code: [] for code in FLORES_LANGUAGE_CODES.values()}

    for split in DEFAULT_SPLITS:
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
            language = CODE_TO_LANGUAGE.get(record_code)
            if language is None:
                continue

            metadata = {"split": split, "flores_code": record_code}
            if isinstance(record.get("id"), str):
                metadata["id"] = record["id"]  # type: ignore[index]

            cache[record_code].append(
                PromptExample(
                    prompt=text.strip(),
                    source="flores_plus",
                    language=language,
                    metadata=metadata,
                )
            )

    _LANGUAGE_PROMPT_CACHE = cache
    return cache

def load_flores_plus(
    language: Language,
    num_samples: int | None = None,
    seed: int | None = None,
) -> List[PromptExample]:
    """Load FLORES++ prompts for a given language across default splits."""

    code = FLORES_LANGUAGE_CODES.get(language)
    if code is None:
        raise ValueError(f"No FLORES++ code registered for language '{language.value}'")

    cache = _ensure_language_cache()
    examples = cache.get(code, [])
    return sample_examples(examples, num_samples)
