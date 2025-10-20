# %%
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import gc

import torch
from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"


# %%
LANGUAGE_CODES: Mapping[str, str] = {
    "English": "eng_Latn",
    "Chinese (Simplified)": "cmn_Hans",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "Portuguese": "por_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Russian": "rus_Cyrl",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Vietnamese": "vie_Latn",
    "Thai": "tha_Thai",
    "Arabic (MSA)": "arb_Arab",
}

LANGUAGE_NAMES: Sequence[str] = tuple(LANGUAGE_CODES.keys())


def collect_parallel_sentences(dataset_dict: DatasetDict) -> Dict[str, List[str]]:
    """Collect sentences for each language across every split."""

    sentences_by_language = {language: [] for language in LANGUAGE_NAMES}
    code_to_language = {code: language for language, code in LANGUAGE_CODES.items()}

    for split_name, split_dataset in dataset_dict.items():
        iterable: Iterable = split_dataset if isinstance(split_dataset, Dataset) else []
        for example in tqdm(
            iterable,
            total=len(split_dataset),
            desc=f"Collecting {split_name}",
        ):
            if not isinstance(example, Mapping):
                continue

            iso_639_3 = example.get("iso_639_3")
            iso_15924 = example.get("iso_15924")
            text = example.get("text")

            if not (isinstance(iso_639_3, str) and isinstance(iso_15924, str) and isinstance(text, str)):
                continue

            language_code = f"{iso_639_3.strip()}_{iso_15924.strip()}"
            language = code_to_language.get(language_code)
            if language:
                sentences_by_language[language].append(text.strip())

    return sentences_by_language


@dataclass
class LayerLanguageMatrices:
    layers: List[torch.Tensor]
    counts: Dict[str, int]


@torch.no_grad()
def compute_layer_language_means(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sentences_by_language: Mapping[str, Sequence[str]],
    batch_size: int = 16,
) -> LayerLanguageMatrices:
    """Generate mean final-token activations for each language across model layers."""

    language_order = [lang for lang in LANGUAGE_NAMES if lang in sentences_by_language]
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    sum_store = {
        language: torch.zeros((num_layers, hidden_size), dtype=torch.float64)
        for language in language_order
    }
    counts = {language: 0 for language in language_order}

    model.eval()

    for language in tqdm(language_order, desc="Computing language activations"):
        sentences = sentences_by_language[language]
        if not sentences:
            continue

        for start in range(0, len(sentences), batch_size):
            batch_sentences = sentences[start : start + batch_size]
            conversations = [
                [{"role": "user", "content": sentence}] for sentence in batch_sentences
            ]
            tokenized = tokenizer.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            outputs = model(
                **tokenized,
                use_cache=False,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[1:]
            attention_mask = tokenized["attention_mask"]
            lengths = attention_mask.sum(dim=1) - 1
            lengths = lengths.clamp(min=0)
            batch_indices = torch.arange(len(batch_sentences), device=model.device)

            for layer_idx, layer_hidden in enumerate(hidden_states):
                final_activations = layer_hidden[batch_indices, lengths]
                sum_store[language][layer_idx] += final_activations.detach().to(
                    "cpu", dtype=torch.float64
                ).sum(dim=0)

            counts[language] += len(batch_sentences)

            del tokenized, outputs, hidden_states
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    layer_matrices: List[torch.Tensor] = []
    for layer_idx in range(num_layers):
        layer_vectors = []
        for language in language_order:
            if counts[language] == 0:
                raise ValueError(f"No examples collected for language '{language}'")
            mean_vector = (sum_store[language][layer_idx] / counts[language]).to(torch.float32)
            layer_vectors.append(mean_vector)
        layer_matrix = torch.stack(layer_vectors, dim=0).T.contiguous()
        layer_matrices.append(layer_matrix)

    return LayerLanguageMatrices(layer_matrices, counts)


# %%
dataset_dict = load_dataset("openlanguagedata/flores_plus")


# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=model_dtype,
).to(device)


# %%
sentences_by_language = collect_parallel_sentences(dataset_dict)


# %%
results = compute_layer_language_means(model, tokenizer, sentences_by_language)


# %%
final_mat = []

for layer_idx, matrix in enumerate(results.layers):
    final_mat.append(matrix)
# %%
final_mat = torch.stack(final_mat)
# %%
torch.save(final_mat, 'output/linguistic-subspace/MdL.pt')
