from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, MutableMapping


class Language(str, Enum):
    """Canonical set of languages used by multilingual datasets."""

    ARABIC = "arabic"
    CHINESE = "chinese"
    ENGLISH = "english"
    FRENCH = "french"
    GERMAN = "german"
    ITALIAN = "italian"
    JAPANESE = "japanese"
    KOREAN = "korean"
    PORTUGUESE = "portuguese"
    RUSSIAN = "russian"
    SPANISH = "spanish"
    THAI = "thai"
    VIETNAMESE = "vietnamese"

    def __str__(self) -> str:
        return self.value


# ISO-639-1 language codes for the supported languages.
LANGUAGE_CODE_MAP: Mapping[Language, str] = {
    Language.ARABIC: "ar",
    Language.CHINESE: "zh",
    Language.ENGLISH: "en",
    Language.FRENCH: "fr",
    Language.GERMAN: "de",
    Language.ITALIAN: "it",
    Language.JAPANESE: "ja",
    Language.KOREAN: "ko",
    Language.PORTUGUESE: "pt",
    Language.RUSSIAN: "ru",
    Language.SPANISH: "es",
    Language.THAI: "th",
    Language.VIETNAMESE: "vi",
}


# FLORES++ language identifiers (ISO-639-3 + ISO-15924).
FLORES_LANGUAGE_CODES: Mapping[Language, str] = {
    Language.ARABIC: "arb_Arab",
    Language.CHINESE: "cmn_Hans",
    Language.ENGLISH: "eng_Latn",
    Language.FRENCH: "fra_Latn",
    Language.GERMAN: "deu_Latn",
    Language.ITALIAN: "ita_Latn",
    Language.JAPANESE: "jpn_Jpan",
    Language.KOREAN: "kor_Hang",
    Language.PORTUGUESE: "por_Latn",
    Language.RUSSIAN: "rus_Cyrl",
    Language.SPANISH: "spa_Latn",
    Language.THAI: "tha_Thai",
    Language.VIETNAMESE: "vie_Latn",
}


@dataclass(frozen=True)
class PromptExample:
    """Normalized representation of a single prompt-driven example."""

    prompt: str
    target: str | None = None
    language: Language | None = None
    source: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")
        if self.target is not None and not isinstance(self.target, str):
            raise TypeError("target must be a string when provided")
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping when provided")

    def with_metadata(self, extra: Mapping[str, Any]) -> "PromptExample":
        """Return a copy with merged metadata."""

        if not extra:
            return self
        combined: MutableMapping[str, Any] = {}
        if isinstance(self.metadata, Mapping):
            combined.update(self.metadata)
        combined.update(extra)
        return PromptExample(
            prompt=self.prompt,
            target=self.target,
            language=self.language,
            source=self.source,
            metadata=dict(combined),
        )
