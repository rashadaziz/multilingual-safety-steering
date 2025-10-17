from __future__ import annotations

from typing import List, Sequence

from transformers import PreTrainedTokenizerBase

from ._utils import DATA_DIR, read_json_records, sample_examples
from .types import PromptExample

DEFAULT_FILE = DATA_DIR / "json" / "alphasteer" / "advbench_train.json"

def _format_chat_prompt(
    tokenizer: PreTrainedTokenizerBase,
    user_message: str,
) -> str:
    conversation = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )


def load_advbench(
    tokenizer: PreTrainedTokenizerBase,
    num_samples: int | None = None,
    seed: int | None = None,
) -> List[PromptExample]:
    """Load the AdvBench dataset and render prompts with reference suffixes."""

    dataset_path = DEFAULT_FILE
    examples: List[PromptExample] = []

    for record in read_json_records(dataset_path):
        query = record.get("query")
        if not isinstance(query, str):
            continue
        suffixes = record.get("reference_responses", ())
        prompt = _format_chat_prompt(tokenizer, query)

        has_suffixes = False
        if isinstance(suffixes, Sequence) and not isinstance(suffixes, (str, bytes)):
            for suffix in suffixes:
                if not isinstance(suffix, str):
                    continue
                has_suffixes = True
                full_prompt = f"{prompt}{suffix}"
                examples.append(
                    PromptExample(
                        prompt=full_prompt.strip(),
                        target=suffix,
                        source="advbench",
                        metadata={"query": query},
                    )
                )

        if not has_suffixes:
            examples.append(
                PromptExample(
                    prompt=prompt.strip(),
                    source="advbench",
                    metadata={"query": query},
                )
            )

    return sample_examples(examples, num_samples, seed)
