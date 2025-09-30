# %%
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from activation_steering import SteeringVector

# %%
def language_subspace_probing(
    language_means: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if language_means.ndim != 2:
        raise ValueError("language_means must be a 2D tensor of shape (d, L)")

    d, num_languages = language_means.shape
    if num_languages == 0:
        raise ValueError("language_means must contain at least one language")
    if rank <= 0:
        raise ValueError("rank must be a positive integer")

    ones = torch.ones(
        num_languages,
        device=language_means.device,
        dtype=language_means.dtype,
    )

    # 1. Approximate M in low rank: language-agnostic mean component
    mean_vector = (language_means @ ones) / num_languages
    M_a_prime = mean_vector.unsqueeze(1)
    residual = language_means - M_a_prime @ ones.unsqueeze(0)

    def _top_r_svd(
        matrix: torch.Tensor, r: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        effective_rank = min(r, S.shape[0])
        if effective_rank == 0:
            zero_left = torch.zeros(matrix.shape[0], 0, device=matrix.device, dtype=matrix.dtype)
            zero_right = torch.zeros(matrix.shape[1], 0, device=matrix.device, dtype=matrix.dtype)
            return zero_left, zero_right, S[:0]
        U_r = U[:, :effective_rank]
        S_r = S[:effective_rank]
        Vh_r = Vh[:effective_rank, :]
        basis = U_r
        coords = Vh_r.T @ torch.diag(S_r)
        return basis, coords, S_r

    # 2. Initial language-specific component via top-r SVD of residual
    M_s_prime, Gamma, _ = _top_r_svd(residual, rank)

    # 3. Reconstruct the approximation to enforce shared + specific structure
    M_prime = M_a_prime @ ones.unsqueeze(0) + M_s_prime @ Gamma.T

    # 4. Force the language-agnostic component to be orthogonal to specific part
    pinv_M_prime = torch.linalg.pinv(M_prime)
    M_a = pinv_M_prime.T @ ones
    norm_sq = torch.dot(M_a, M_a)
    if torch.isnan(norm_sq) or norm_sq <= 0:
        raise ValueError("Unable to normalize language-agnostic component (zero norm)")
    M_a = (M_a / norm_sq).unsqueeze(1)

    # 5. Top-r SVD on the orthogonalized residual for the final specific subspace
    orthogonal_residual = M_prime - M_a @ ones.unsqueeze(0)
    M_s, Gamma, _ = _top_r_svd(orthogonal_residual, rank)

    return M_a.contiguous(), M_s.contiguous(), Gamma.contiguous()


# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

refusal_vector_path = "output/cast/refusal_vectors.svec"
language_means_path = "output/linguistic-subspace/MdL.pt"
dataset_path = "data/eval/harmful_test_translated_ko.json"
output_path = "output/cast/korean_linguistic_subspace.json"

# Steering / ablation hyperparameters
linguistic_layers = [19, 20, 21, 22, 23, 24, 25, 26, 27]
refusal_layers = [10, 11, 12, 13, 14, 15]
refusal_strength = 2.5
subspace_rank = 4
middle_layer_lambda = 0.0  # λ for mid-stack layers (0 → 0.4 in paper)
higher_layer_lambda = -0.02  # λ for upper layers (−0.4 → 0 in paper)


# %%
def load_dataset_prompts(path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [item["instruction_translated"] for item in data]
    return prompts, data


def prepare_language_subspaces(
    means_tensor: torch.Tensor,
    rank: int,
) -> Dict[int, torch.Tensor]:
    subspaces: Dict[int, torch.Tensor] = {}
    for layer_idx in range(means_tensor.shape[0]):
        layer_means = means_tensor[layer_idx].to(torch.float32)
        _, layer_basis, _ = language_subspace_probing(layer_means, rank)
        if layer_basis.numel() == 0:
            continue
        subspaces[layer_idx] = layer_basis.contiguous()
    return subspaces


def build_combined_hook(
    basis: torch.Tensor | None,
    refusal_vector: torch.Tensor | None,
    lambda_value: float,
    refusal_scale: float,
) -> Callable[[torch.nn.Module, Tuple[Any, ...], torch.Tensor], torch.Tensor]:
    basis = basis.to(device) if basis is not None else None
    refusal_vector = refusal_vector.to(device) if refusal_vector is not None else None

    def _hook(_module, _input, output):
        hidden = output
        original_dtype = hidden.dtype
        hidden = hidden.to(torch.float32)

        if refusal_vector is not None and refusal_scale != 0.0:
            hidden = hidden + refusal_scale * refusal_vector

        # We skip linguistic projection during generation
        if _input[0].shape[1] == 1:
            return hidden.to(original_dtype)

        if basis is not None and lambda_value != 0.0:
            last = hidden[:, -1, :]
            coords = torch.matmul(last, basis)
            projection = torch.matmul(coords, basis.T)
            hidden[:, -1, :] = last - lambda_value * projection

        return hidden.to(original_dtype)

    return _hook


# %%
prompts, raw_records = load_dataset_prompts(dataset_path)


# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
generation_config = GenerationConfig.from_pretrained(model_name)


# %%
means_tensor = torch.load(language_means_path, map_location="cpu")
language_subspaces = prepare_language_subspaces(means_tensor, subspace_rank)

steering_vector = SteeringVector.load(refusal_vector_path)


# %%
handles = []

linguistic_layers_sorted = sorted(set(linguistic_layers))
linguistic_layers_set = set(linguistic_layers_sorted)
refusal_layers_set = set(refusal_layers)
all_layers = sorted(linguistic_layers_set | refusal_layers_set)

middle_layer_set = []
higher_layer_set = linguistic_layers

for idx, layer in enumerate(model.model.layers):
    if idx not in all_layers:
        continue

    lambda_value = 0.0
    if idx in middle_layer_set:
        lambda_value = middle_layer_lambda
    elif idx in higher_layer_set:
        lambda_value = higher_layer_lambda

    basis = language_subspaces.get(idx) if idx in linguistic_layers_set else None

    refusal_tensor = None
    layer_refusal_scale = 0.0
    if idx in refusal_layers_set and idx < len(steering_vector.directions):
        refusal_tensor = torch.tensor(steering_vector.directions[idx], dtype=torch.float32)
        layer_refusal_scale = refusal_strength

    hook = build_combined_hook(
        basis,
        refusal_tensor,
        lambda_value,
        layer_refusal_scale,
    )
    handles.append(layer.register_forward_hook(hook))


# %%
responses: List[Dict[str, Any]] = []

batch_size = 32
num_batches = (len(prompts) + batch_size - 1) // batch_size if prompts else 0

for start in tqdm(
    range(0, len(prompts), batch_size),
    total=num_batches,
    desc="Steered generation",
):
    batch_prompts = prompts[start : start + batch_size]
    conversations = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]

    inputs = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        generation_config=generation_config,
        max_new_tokens=128,
        do_sample=False,
    )

    generated_sequences = outputs[:, inputs["input_ids"].shape[1] :].cpu()
    decoded = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

    originals = raw_records[start : start + len(batch_prompts)]
    for prompt, response_text, record in zip(batch_prompts, decoded, originals):
        entry: Dict[str, Any] = {
            "prompt": prompt,
            "response": response_text.strip(),
        }
        for key in ("id", "label", "language"):
            if key in record:
                entry[key] = record[key]
        responses.append(entry)

    del inputs, outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


for handle in handles:
    handle.remove()


Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(responses, f, ensure_ascii=False, indent=2)

print(f"Saved steered generations to {output_path}")

# %%
