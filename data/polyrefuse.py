from __future__ import annotations

from typing import Dict, List, Literal

from ._utils import DATA_DIR, read_json_records, sample_examples
from .types import Language, PromptExample

DEFAULT_DIR = DATA_DIR / "json" / "polyrefuse"
HARMLESS_DIR = DATA_DIR / "json" / "polyrefuse_harmless"

_FILE_MAP: Dict[Language, str] = {
    Language.ARABIC: "harmful_test_translated_ar.json",
    Language.GERMAN: "harmful_test_translated_de.json",
    Language.ENGLISH: "harmful_test_translated_en.json",
    Language.SPANISH: "harmful_test_translated_es.json",
    Language.FRENCH: "harmful_test_translated_fr.json",
    Language.ITALIAN: "harmful_test_translated_it.json",
    Language.JAPANESE: "harmful_test_translated_ja.json",
    Language.KOREAN: "harmful_test_translated_ko.json",
    Language.RUSSIAN: "harmful_test_translated_ru.json",
    Language.THAI: "harmful_test_translated_th.json",
    Language.CHINESE: "harmful_test_translated_zh.json",
}

_FILE_MAP_HARMLESS: Dict[Language, str] = {
    Language.ENGLISH: "harmless_test_translated_en.json",
    Language.KOREAN: "harmless_test_translated_ko.json",
}


def available_polyrefuse_languages() -> List[Language]:
    """Return the languages that have PolyRefuse translations available."""

    return sorted(_FILE_MAP.keys(), key=lambda language: language.value)


def load_polyrefuse(
    language: Language,
    kind: Literal['harmful', 'harmless'] = 'harmful',
    num_samples: int | None = None,
    seed: int | None = None,
) -> List[PromptExample]:
    """Load PolyRefuse prompts for a specific language."""

    try:
        if kind == 'harmful':
            file_name = _FILE_MAP[language]
        else:
            file_name = _FILE_MAP_HARMLESS[language]
    except KeyError as exc:
        raise ValueError(f"No PolyRefuse split available for language '{language.value}'") from exc

    if kind == 'harmful':
        dataset_path = DEFAULT_DIR / file_name
    else:
        dataset_path = HARMLESS_DIR / file_name

    examples: List[PromptExample] = []
    for record in read_json_records(dataset_path):
        instruction = record.get("instruction")
        if not isinstance(instruction, str):
            continue
        category = record.get("category")
        metadata = {"category": category} if isinstance(category, str) else None
        examples.append(
            PromptExample(
                prompt=instruction.strip(),
                source="polyrefuse",
                language=language,
                metadata=metadata,
            )
        )

    return sample_examples(examples, num_samples, seed)
