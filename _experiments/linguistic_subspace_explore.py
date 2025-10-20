from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from activation_steering import SteeringVector
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


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


@dataclass
class PromptRecord:
    text: str
    label: str


def _principal_angle(vector: torch.Tensor, basis: torch.Tensor) -> float | None:
    if vector is None or basis is None or vector.numel() == 0 or basis.numel() == 0:
        return None

    flat_vector = vector.reshape(-1)
    vector_norm = torch.linalg.norm(flat_vector).item()
    if not math.isfinite(vector_norm) or vector_norm == 0.0:
        return None

    coords = flat_vector @ basis
    projection_norm = torch.linalg.norm(coords).item()
    cos_theta = projection_norm / vector_norm
    if not math.isfinite(cos_theta):
        return None

    cos_theta = max(min(cos_theta, 1.0), -1.0)
    return math.acos(cos_theta)


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

    M_s_prime, Gamma, _ = _top_r_svd(residual, rank)

    M_prime = M_a_prime @ ones.unsqueeze(0) + M_s_prime @ Gamma.T

    pinv_M_prime = torch.linalg.pinv(M_prime)
    M_a = pinv_M_prime.T @ ones
    norm_sq = torch.dot(M_a, M_a)
    if torch.isnan(norm_sq) or norm_sq <= 0:
        raise ValueError("Unable to normalize language-agnostic component (zero norm)")
    M_a = (M_a / norm_sq).unsqueeze(1)

    orthogonal_residual = M_prime - M_a @ ones.unsqueeze(0)
    M_s, Gamma, _ = _top_r_svd(orthogonal_residual, rank)

    return M_a.contiguous(), M_s.contiguous(), Gamma.contiguous()


def collect_parallel_sentences(dataset_dict: DatasetDict) -> Dict[str, List[str]]:
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


def _sanitize_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def load_flores_prompts(
    language_names: Sequence[str],
    splits: Sequence[str],
    samples_per_language: int | None = None,
) -> List[PromptRecord]:
    dataset = load_dataset("openlanguagedata/flores_plus")
    selected_splits = {
        split: dataset[split] for split in splits if isinstance(dataset, MutableMapping) and split in dataset
    }
    sentences_by_language = collect_parallel_sentences(DatasetDict(selected_splits))

    records: List[PromptRecord] = []
    for language in language_names:
        sentences = sentences_by_language.get(language, [])
        if samples_per_language is not None:
            sentences = sentences[:samples_per_language]
        records.extend(PromptRecord(text=sentence, label=language) for sentence in sentences)
    return records


def load_local_prompts(
    paths: Sequence[Path],
    samples_per_language: int | None = None,
    text_field: str = "instruction_translated",
    fallback_fields: Sequence[str] = ("instruction", "prompt", "text"),
) -> List[PromptRecord]:
    records: List[PromptRecord] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        collected: List[str] = []
        for example in data:
            if not isinstance(example, Mapping):
                continue
            text = example.get(text_field)
            if not isinstance(text, str):
                for field in fallback_fields:
                    candidate = example.get(field)
                    if isinstance(candidate, str):
                        text = candidate
                        break
            if not isinstance(text, str):
                continue
            collected.append(text.strip())
            if samples_per_language is not None and len(collected) >= samples_per_language:
                break
        label = path.stem
        records.extend(PromptRecord(text=sentence, label=label) for sentence in collected)
    return records


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


def collect_original_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records: Sequence[PromptRecord],
    batch_size: int,
) -> Dict[int, List[torch.Tensor]]:
    layer_cache: Dict[int, List[torch.Tensor]] = {
        idx: [] for idx in range(model.config.num_hidden_layers)
    }

    for start in tqdm(
        range(0, len(records), batch_size),
        total=(len(records) + batch_size - 1) // batch_size if records else 0,
        desc="Collecting activations",
    ):
        batch = records[start : start + batch_size]
        conversations = [
            [{"role": "user", "content": item.text}] for item in batch
        ]
        inputs = tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                use_cache=False,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[1:]
        attention_mask = inputs["attention_mask"]
        lengths = attention_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        batch_indices = torch.arange(attention_mask.shape[0], device=model.device)

        for layer_idx, layer_hidden in enumerate(hidden_states):
            final_activations = layer_hidden[batch_indices, lengths]
            layer_cache[layer_idx].append(final_activations.detach().to("cpu"))

        del inputs, outputs, hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return layer_cache


def collect_intervened_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records: Sequence[PromptRecord],
    batch_size: int,
    *,
    apply_language: bool,
    language_bases: Mapping[int, torch.Tensor] | None,
    language_strength: float,
    apply_refusal: bool,
    refusal_vectors: Mapping[int, torch.Tensor] | None,
    refusal_strength: float,
    independent_layers: bool,
) -> Dict[int, List[torch.Tensor]]:
    layer_cache: Dict[int, List[torch.Tensor]] = {
        idx: [] for idx in range(model.config.num_hidden_layers)
    }

    device = model.device
    bases_on_device: Dict[int, torch.Tensor] = {}
    bases_on_device = {idx: basis.to(device) for idx, basis in language_bases.items()}

    refusal_on_device: Dict[int, torch.Tensor] = {}
    if apply_refusal and refusal_vectors is not None:
        refusal_on_device = {idx: vec.to(device) for idx, vec in refusal_vectors.items()}

    context: Dict[str, torch.Tensor | None] = {
        "lengths": None,
        "batch_indices": None,
    }

    handles = []
    for layer_idx, layer_module in enumerate(model.model.layers):
        layer_basis = bases_on_device.get(layer_idx) if bases_on_device else None
        layer_refusal = refusal_on_device.get(layer_idx) if refusal_on_device else None

        def _make_hook(
            idx: int,
            basis: torch.Tensor | None,
            refusal: torch.Tensor | None,
        ):
            refusal_delta: torch.Tensor | None = None
            if apply_refusal and refusal is not None and refusal_strength != 0.0:
                if basis is None or basis.numel() == 0:
                    logger.warning(
                        "Layer %d: refusal vector provided without a language basis; skipping principal angle logging",
                        idx,
                    )
                else:
                    coords = refusal @ basis
                    projection = coords @ basis.T
                    refusal_delta = refusal - 0.4 * projection

                    original_angle = _principal_angle(refusal, basis)
                    projected_angle = _principal_angle(refusal_delta, basis)
                    if original_angle is not None and projected_angle is not None:
                        logger.info(
                            "Layer %d principal angles (rad/deg) - refusal: %.4f/%.2f°, refusal_proj: %.4f/%.2f°",
                            idx,
                            original_angle,
                            math.degrees(original_angle),
                            projected_angle,
                            math.degrees(projected_angle),
                        )

            def _hook(_module, _input, output):
                lengths = context["lengths"]
                batch_indices = context["batch_indices"]
                if lengths is None or batch_indices is None:
                    raise RuntimeError("Batch context lengths unavailable")

                hidden = output
                original_dtype = hidden.dtype
                hidden_fp32 = hidden.to(torch.float32)
                working = hidden_fp32.clone() if independent_layers else hidden_fp32

                if apply_refusal and refusal_delta is not None and refusal_strength != 0.0:
                    working = working + refusal_strength * refusal_delta

                final = working[batch_indices, lengths]

                if apply_language and basis is not None and language_strength != 0.0:
                    coeffs = final @ basis
                    projection = coeffs @ basis.T
                    final = final - language_strength * projection

                working[batch_indices, lengths] = final

                layer_cache[idx].append(final.detach().to("cpu"))
                if independent_layers:
                    return hidden
                return working.to(original_dtype)

            return _hook

        handles.append(
            layer_module.register_forward_hook(
                _make_hook(layer_idx, layer_basis, layer_refusal)
            )
        )

    for start in tqdm(
        range(0, len(records), batch_size),
        total=(len(records) + batch_size - 1) // batch_size if records else 0,
        desc="Collecting intervened activations",
    ):
        batch = records[start : start + batch_size]
        conversations = [
            [{"role": "user", "content": item.text}] for item in batch
        ]
        inputs = tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        attention_mask = inputs["attention_mask"]
        lengths = attention_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0).to(device)
        batch_size_actual = attention_mask.shape[0]
        batch_indices = torch.arange(batch_size_actual, device=device)

        context["lengths"] = lengths
        context["batch_indices"] = batch_indices

        with torch.no_grad():
            model(
                **inputs,
                use_cache=False,
                output_hidden_states=False,
            )

        context["lengths"] = None
        context["batch_indices"] = None

        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for handle in handles:
        handle.remove()

    return layer_cache


def stack_layer_cache(layer_cache: Mapping[int, List[torch.Tensor]]) -> Dict[int, torch.Tensor]:
    stacked: Dict[int, torch.Tensor] = {}
    for layer_idx, tensors in layer_cache.items():
        if not tensors:
            continue
        stacked[layer_idx] = torch.cat(tensors, dim=0)
    return stacked


def compute_pca_projection(
    activations: torch.Tensor,
    n_components: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if activations.ndim != 2:
        raise ValueError("activations must be a 2D tensor")
    if activations.shape[0] < n_components:
        raise ValueError("PCA requires at least as many samples as components")

    activations_np = activations.detach().float().cpu().numpy()
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(activations_np)
    components = torch.from_numpy(transformed.astype(np.float32))
    explained_ratio = torch.from_numpy(pca.explained_variance_ratio_.astype(np.float32))

    return components, explained_ratio


def plot_layer_pca(
    layer_idx: int,
    original_proj: torch.Tensor,
    ablated_proj: torch.Tensor,
    labels: Sequence[str],
    original_ratio: torch.Tensor,
    ablated_ratio: torch.Tensor,
    output_path: Path,
) -> None:
    if original_proj.shape[0] != ablated_proj.shape[0] or original_proj.shape[0] != len(labels):
        raise ValueError("Projection counts must match label count")

    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    color_map = {label: cmap(idx % cmap.N) for idx, label in enumerate(unique_labels)}

    original_np = original_proj.cpu().numpy()
    ablated_np = ablated_proj.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    for ax, data, ratios, title in (
        (axes[0], original_np, original_ratio, "Original"),
        (axes[1], ablated_np, ablated_ratio, "Intervened"),
    ):
        for label in unique_labels:
            mask = np.array([lbl == label for lbl in labels])
            if not mask.any():
                continue
            ax.scatter(
                data[mask, 0],
                data[mask, 1],
                label=label,
                s=12,
                alpha=0.7,
                color=color_map[label],
            )
        pc1 = ratios[0].item() * 100 if ratios.numel() > 0 else 0.0
        pc2 = ratios[1].item() * 100 if ratios.numel() > 1 else 0.0
        ax.set_title(f"{title} – PC1 {pc1:.1f}% | PC2 {pc2:.1f}%")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, linestyle="--", alpha=0.2)

    axes[0].legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize="small")
    fig.suptitle(f"Layer {layer_idx} final-token PCA")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize PCA before/after language subspace ablation")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--language-means", type=str, default="output/linguistic-subspace/MdL.pt")
    parser.add_argument("--subspace-rank", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dataset-type", choices=("flores_plus", "local"), default="local")
    parser.add_argument("--flores-languages", nargs="+", default=None)
    parser.add_argument("--flores-splits", nargs="+", default=("dev",))
    parser.add_argument("--local-paths", nargs="+", default=None)
    parser.add_argument("--local-dir", type=str, default="data/eval")
    parser.add_argument("--local-text-field", type=str, default="instruction_translated")
    parser.add_argument(
        "--local-fallback-fields",
        nargs="+",
        default=("instruction", "prompt", "text"),
    )
    parser.add_argument("--samples-per-language", type=int, default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="output/linguistic-subspace/pca")
    parser.add_argument("--intervention", choices=("language", "refusal", "both"), default="language")
    parser.add_argument("--refusal-vector-path", type=str, default='output/cast/refusal_vectors.svec')
    parser.add_argument("--refusal-strength", type=float, default=2.5)
    parser.add_argument("--refusal-layers", nargs="+", type=int, default=None)
    parser.add_argument("--language-strength", type=float, default=1.0)
    parser.add_argument(
        "--independent-layers",
        action="store_true",
        help="Record interventions per layer without feeding modifications forward",
    )
    return parser.parse_args()


def load_records_from_args(args: argparse.Namespace) -> Tuple[List[PromptRecord], str]:
    if args.dataset_type == "flores_plus":
        languages = args.flores_languages if args.flores_languages else list(LANGUAGE_NAMES)
        splits = list(args.flores_splits) if args.flores_splits else ["dev"]
        records = load_flores_prompts(languages, splits, args.samples_per_language)
        language_tag = "-".join(_sanitize_tag(lang) for lang in sorted(set(languages)))
        dataset_tag = "flores" if not language_tag else f"flores_{language_tag}"
        return records, dataset_tag

    paths: List[Path]
    if args.local_paths:
        paths = [Path(p) for p in args.local_paths]
    else:
        base_dir = Path(args.local_dir)
        paths = sorted(base_dir.glob("*.json"))
    if not paths:
        raise ValueError("No local dataset files found")

    fallback_fields = tuple(args.local_fallback_fields) if args.local_fallback_fields else ()
    records = load_local_prompts(
        paths,
        samples_per_language=args.samples_per_language,
        text_field=args.local_text_field,
        fallback_fields=fallback_fields,
    )
    if args.local_paths:
        stems = [_sanitize_tag(path.stem) for path in paths]
        joined = "-".join(stems[:3])
        if len(stems) > 3:
            joined = f"{joined}_etc"
    else:
        joined = _sanitize_tag(Path(args.local_dir).name or "local")
    dataset_tag = f"local_{joined}" if joined else "local"
    return records, dataset_tag


def main() -> None:
    args = parse_args()

    records, dataset_tag = load_records_from_args(args)
    if not records:
        raise ValueError("No prompts loaded")

    if args.max_prompts is not None and args.max_prompts > 0:
        records = records[: args.max_prompts]

    labels = [record.label for record in records]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=model_dtype,
    ).to(device)
    model.eval()

    intervention = args.intervention
    apply_language = intervention in {"language", "both"}
    apply_refusal = intervention in {"refusal", "both"}

    language_bases: Dict[int, torch.Tensor] | None = None
    language_strength = float(args.language_strength)
    means_tensor = torch.load(args.language_means, map_location="cpu")
    language_bases = prepare_language_subspaces(means_tensor, args.subspace_rank)

    refusal_vectors: Dict[int, torch.Tensor] | None = None
    refusal_strength = float(args.refusal_strength)
    if apply_refusal:
        if not args.refusal_vector_path:
            raise ValueError("--refusal-vector-path is required when using refusal interventions")
        steering_vector = SteeringVector.load(args.refusal_vector_path)
        available = len(steering_vector.directions)
        if args.refusal_layers:
            layer_ids = sorted(set(int(layer) for layer in args.refusal_layers if layer >= 0))
        else:
            layer_ids = list(range(available))
        refusal_vectors = {}
        for layer_idx in layer_ids:
            if layer_idx >= available:
                continue
            tensor = torch.tensor(steering_vector.directions[layer_idx], dtype=torch.float32)
            refusal_vectors[layer_idx] = tensor
        if not refusal_vectors:
            raise ValueError("No refusal vectors available for the specified layers")

    original_cache = collect_original_activations(model, tokenizer, records, args.batch_size)

    independent_layers = bool(args.independent_layers)

    modified_cache = collect_intervened_activations(
        model,
        tokenizer,
        records,
        args.batch_size,
        apply_language=apply_language,
        language_bases=language_bases,
        language_strength=language_strength,
        apply_refusal=apply_refusal,
        refusal_vectors=refusal_vectors,
        refusal_strength=refusal_strength,
        independent_layers=independent_layers,
    )

    original_layers = stack_layer_cache(original_cache)
    modified_layers = stack_layer_cache(modified_cache)

    if args.layers:
        target_layers = sorted(set(args.layers))
    else:
        target_layers = sorted(original_layers.keys())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag_prefix = f"{dataset_tag}_{intervention}"

    for layer_idx in target_layers:
        original_tensor = original_layers.get(layer_idx)
        ablated_tensor = modified_layers.get(layer_idx)
        if original_tensor is None or ablated_tensor is None:
            print(f"Skipping layer {layer_idx}: missing activations")
            continue
        if original_tensor.shape[0] != len(labels) or ablated_tensor.shape[0] != len(labels):
            print(f"Skipping layer {layer_idx}: activation counts do not match prompts")
            continue

        try:
            original_proj, original_ratio = compute_pca_projection(original_tensor)
            ablated_proj, ablated_ratio = compute_pca_projection(ablated_tensor)
        except ValueError as exc:
            print(f"Skipping layer {layer_idx}: {exc}")
            continue

        plot_path = output_dir / f"{tag_prefix}_layer_{layer_idx:02d}_pca.png"
        plot_layer_pca(
            layer_idx,
            original_proj,
            ablated_proj,
            labels,
            original_ratio,
            ablated_ratio,
            plot_path,
        )
        print(f"Saved {plot_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
