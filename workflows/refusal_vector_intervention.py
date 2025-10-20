import argparse
import json
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from activation_steering import SteeringVector
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from core.steerable import SteerableModel
from data import Language, PromptExample, load_polyrefuse
from hooks.refusal import build_refusal_hook
from utils.language_subspace import compute_language_subspace


def _parse_language(value: str) -> Language:
    try:
        return Language(value.lower())
    except ValueError as exc:
        valid = ", ".join(lang.value for lang in Language)
        raise argparse.ArgumentTypeError(f"Unknown language '{value}'. Choose from: {valid}") from exc


def _load_language_projection(means_path: Path, rank: int) -> Dict[int, torch.Tensor]:
    language_means = torch.load(means_path, map_location="cpu")
    projection: Dict[int, torch.Tensor] = {}

    if isinstance(language_means, dict):
        # Assume mapping of layer index -> basis tensor.
        for key, value in language_means.items():
            basis = torch.as_tensor(value, dtype=torch.float32)
            if basis.ndim == 1:
                basis = basis.unsqueeze(1)
            projection[int(key)] = basis.contiguous()
        return projection

    if not isinstance(language_means, torch.Tensor):
        raise TypeError("Language means file must be a torch.Tensor or mapping of layers to tensors.")

    if language_means.ndim == 2:
        language_means = language_means.unsqueeze(0)

    if language_means.ndim != 3:
        raise ValueError("Expected language means tensor of shape (num_layers, hidden_size, num_languages).")

    for layer_idx in range(language_means.shape[0]):
        layer_means = language_means[layer_idx]
        if layer_means.numel() == 0:
            continue
        basis, _ = compute_language_subspace(layer_means, rank)
        if basis.numel() == 0:
            continue
        projection[layer_idx] = basis.to(dtype=torch.float32).contiguous()

    return projection


def _chunked(sequence: Sequence[PromptExample], size: int) -> Iterable[Sequence[PromptExample]]:
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


def _parse_layers(values: Sequence[str]) -> List[int]:
    parsed: List[int] = []
    for value in values:
        parts = value.split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            try:
                parsed.append(int(part))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"Invalid layer index '{part}'. Expected integers.") from exc
    return parsed


def _register_refusal_hooks(
    steerable: SteerableModel,
    refusal_vector: SteeringVector,
    *,
    vector_strength: float,
    projection: Dict[int, torch.Tensor] | None,
    projection_strength: float,
    layers: Sequence[int] | None,
) -> None:
    if layers:
        target_layers = list(dict.fromkeys(layers))
    else:
        target_layers = list(range(min(len(steerable.layers), len(refusal_vector.directions))))

    for layer_idx in target_layers:
        if layer_idx >= len(refusal_vector.directions) or layer_idx >= len(steerable.layers):
            continue
        direction = refusal_vector.directions[layer_idx]
        if direction is None:
            continue
        direction_tensor = torch.tensor(direction, dtype=torch.float32)
        basis = projection.get(layer_idx) if projection is not None else None
        hook = build_refusal_hook(
            direction_tensor,
            vector_strength=vector_strength,
            ablation_projection=basis,
            ablation_strength=projection_strength,
        )
        steerable.register_hook(layer_idx, hook)


def _prompt_to_messages(examples: Sequence[PromptExample]) -> List[List[dict]]:
    conversations: List[List[dict]] = []
    for example in examples:
        conversations.append(
            [{"role": "user", "content": example.prompt}]
        )
    return conversations


def _generate_split(
    *,
    split: str,
    output_path: Path,
    language: Language,
    use_translated_text: bool,
    num_prompts: int,
    seed: int,
    batch_size: int,
    steerable: SteerableModel,
    generation_kwargs: dict,
    steering_label: str,
) -> int:
    try:
        prompts = load_polyrefuse(
            language,
            kind=split,
            use_translated_text=use_translated_text,
            num_samples=num_prompts,
            seed=seed,
        )
    except ValueError as exc:
        print(f"Skipping {split} split: {exc}")
        return 0

    if not prompts:
        print(f"No prompts available for the {split} split.")
        return 0

    results: List[dict] = []

    num_batches = ceil(len(prompts) / batch_size) if prompts else 0

    for batch in tqdm(
        _chunked(prompts, batch_size),
        total=num_batches,
        desc=f"Generating responses ({split})",
    ):
        messages = _prompt_to_messages(batch)
        continuations = steerable.generate(
            messages=messages,
            **generation_kwargs,
        )
        for example, response in zip(batch, continuations):
            record = {
                "prompt": example.prompt,
                "response": response.strip(),
                "language": example.language.value if example.language is not None else None,
                "source": example.source,
                "metadata": dict(example.metadata) if example.metadata is not None else None,
                "steering": steering_label,
                "split": split,
            }
            results.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)

    print(f"Wrote {len(results)} prompt-response pairs to {output_path}")
    return len(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate responses with a refusal steering vector.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--refusal-vector", type=Path, default=Path("output/cast/refusal_vector.svec"))
    parser.add_argument("--language", type=_parse_language, default=Language.ENGLISH)
    parser.add_argument("--split", choices=("harmful", "harmless"), default="harmful")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path for the primary split output JSON. Defaults depend on the chosen split.",
    )
    parser.add_argument(
        "--harmless-output-path",
        type=Path,
        default=None,
        help="Where to write responses for the harmless PolyRefuse split when generated.",
    )
    parser.add_argument(
        "--skip-harmless",
        action="store_true",
        help="Skip generating responses for the harmless PolyRefuse split.",
    )
    parser.add_argument("--refusal-strength", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--language-means-path", type=Path, default=None)
    parser.add_argument("--language-rank", type=int, default=4)
    parser.add_argument("--projection-strength", type=float, default=1.0)
    parser.add_argument(
        "--layers",
        nargs="*",
        default=None,
        help="Layer indices (comma separated or space separated) to apply the refusal intervention on.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
    steerable = SteerableModel(model, tokenizer, device)

    steering_vector = SteeringVector.load(str(args.refusal_vector))

    projection = None
    if args.language_means_path is not None:
        projection = _load_language_projection(args.language_means_path, args.language_rank)

    _register_refusal_hooks(
        steerable,
        steering_vector,
        vector_strength=args.refusal_strength,
        projection=projection,
        projection_strength=args.projection_strength,
        layers=_parse_layers(args.layers) if args.layers is not None else None,
    )

    if args.output_path is None:
        default_name = "prompt_responses_harmless.json" if args.split == "harmless" else "prompt_responses.json"
        args.output_path = Path("output/language-refusal") / default_name

    if args.harmless_output_path is None:
        args.harmless_output_path = Path("output/language-refusal/prompt_responses_harmless.json")

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})

    steering_label = "refusal_language_ablated" if projection else "refusal_only"
    use_translated_text = args.language != Language.ENGLISH

    _generate_split(
        split=args.split,
        output_path=args.output_path,
        language=args.language,
        use_translated_text=use_translated_text,
        num_prompts=args.num_prompts,
        seed=args.seed,
        batch_size=args.batch_size,
        steerable=steerable,
        generation_kwargs=generation_kwargs,
        steering_label=steering_label,
    )

    if not args.skip_harmless and args.split != "harmless":
        _generate_split(
            split="harmless",
            output_path=args.harmless_output_path,
            language=args.language,
            use_translated_text=use_translated_text,
            num_prompts=args.num_prompts,
            seed=args.seed,
            batch_size=args.batch_size,
            steerable=steerable,
            generation_kwargs=generation_kwargs,
            steering_label=steering_label,
        )

if __name__ == '__main__':
    main()
