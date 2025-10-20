import argparse
import asyncio
import json
from dotenv import load_dotenv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import fasttext

from fasttext.FastText import _FastText
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from data import FLORES_LANGUAGE_CODES, Language
from utils.external_llm_client import ExternalLLMClient
from utils.harmfulness_judge import HarmfulnessJudgement, judge_prompt_response_harmfulness

load_dotenv()

EXPECTED_GLOT_LABELS: Dict[Language, str] = {
    language: f"__label__{code}"
    for language, code in FLORES_LANGUAGE_CODES.items()
}


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of prompt-response records in {path}")
    return data


def _load_glotlid_model(cache_dir: Path | None = None) -> _FastText:
    model_path = hf_hub_download(
        repo_id="cis-lmu/glotlid",
        filename="model.bin",
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return fasttext.load_model(model_path)


def _predict_language(model: _FastText, text: str) -> Tuple[str | None, float]:
    cleaned = " ".join(text.split())
    if not cleaned:
        cleaned = " "
    labels, scores = model.predict(cleaned)
    if not labels:
        return None, 0.0
    label = labels[0]
    score = float(scores[0]) if scores else 0.0
    return label, score


async def _evaluate_harmfulness_async(
    records: List[Dict[str, Any]],
    *,
    client: ExternalLLMClient,
    model_name: str,
    max_concurrency: int,
) -> Tuple[List[HarmfulnessJudgement], int]:
    max_concurrency = max(1, max_concurrency)
    semaphore = asyncio.Semaphore(max_concurrency)
    results: List[HarmfulnessJudgement | None] = [None] * len(records)
    harmful_count = 0

    progress = tqdm(total=len(records), desc="Judging harmfulness")

    async def _classify(idx: int, record: Dict[str, Any]) -> Tuple[int, HarmfulnessJudgement]:
        async with semaphore:
            judgement = await asyncio.to_thread(
                judge_prompt_response_harmfulness,
                prompt=str(record.get("prompt", "")),
                response=str(record.get("response", "")),
                client=client,
                model=model_name,
            )
        return idx, judgement

    tasks = [asyncio.create_task(_classify(idx, record)) for idx, record in enumerate(records)]

    try:
        for task in asyncio.as_completed(tasks):
            idx, judgement = await task
            results[idx] = judgement
            if judgement.is_harmful:
                harmful_count += 1
            progress.update(1)
    finally:
        progress.close()

    if any(j is None for j in results):
        raise RuntimeError("Missing harmfulness judgements for one or more records.")

    judgements = cast(List[HarmfulnessJudgement], results)
    return judgements, harmful_count


def _compute_language_fidelity(
    records: List[Dict[str, Any]],
    *,
    model: _FastText,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    per_language_total: Dict[str, int] = defaultdict(int)
    per_language_matches: Dict[str, int] = defaultdict(int)
    language_evaluable = 0
    language_matches = 0
    skipped_examples = 0
    detailed_predictions: List[Dict[str, Any]] = []

    for record in tqdm(records, desc="Detecting response language"):
        response = str(record.get("response", ""))
        predicted_label, score = _predict_language(model, response)
        language_value = record.get("language")

        match: bool | None = None
        expected_label: str | None = None

        if isinstance(language_value, str):
            try:
                language_enum = Language(language_value.lower())
            except ValueError:
                language_enum = None
        else:
            language_enum = None

        if language_enum is not None:
            expected_label = EXPECTED_GLOT_LABELS.get(language_enum)
            if predicted_label is not None and expected_label is not None:
                language_evaluable += 1
                per_language_total[language_enum.value] += 1
                match = predicted_label == expected_label
                if match:
                    language_matches += 1
                    per_language_matches[language_enum.value] += 1
            elif expected_label is not None:
                per_language_total[language_enum.value] += 1
                language_evaluable += 1
                match = False
        else:
            skipped_examples += 1

        detailed_predictions.append(
            {
                "predicted_label": predicted_label,
                "predicted_score": score,
                "expected_label": expected_label,
                "language_match": match,
            }
        )

    fidelity = (language_matches / language_evaluable) if language_evaluable else 0.0
    per_language_stats = {
        language: {
            "total": count,
            "match_count": per_language_matches.get(language, 0),
            "language_fidelity": (
                per_language_matches.get(language, 0) / count if count else 0.0
            ),
        }
        for language, count in per_language_total.items()
    }

    summary = {
        "language_evaluable_examples": language_evaluable,
        "language_match_count": language_matches,
        "language_fidelity": fidelity,
        "per_language": per_language_stats,
        "skipped_examples_without_language": skipped_examples,
    }
    return detailed_predictions, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate refusal intervention outputs.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("output/language-refusal/prompt_responses.json"),
        help="Path to the prompt-response JSON generated by refusal_vector_intervention.py.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("output/language-refusal/evaluation_summary.json"),
        help="Where to write aggregate evaluation metrics.",
    )
    parser.add_argument(
        "--harmfulness-model",
        default="gpt-5",
        help="External judge model identifier.",
    )
    parser.add_argument(
        "--glotlid-cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for the GlotLID model.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Maximum number of concurrent harmfulness judgements.",
    )

    args = parser.parse_args()

    records = _load_json(args.input_path)
    if not records:
        print(f"No records found in {args.input_path}.")
        return

    client = ExternalLLMClient(model=args.harmfulness_model)
    judgements, harmful_count = asyncio.run(
        _evaluate_harmfulness_async(
            records,
            client=client,
            model_name=args.harmfulness_model,
            max_concurrency=args.max_concurrency,
        )
    )

    glotlid_model = _load_glotlid_model(cache_dir=args.glotlid_cache_dir)
    language_details, language_summary = _compute_language_fidelity(
        records,
        model=glotlid_model,
    )

    total_examples = len(records)
    harmful_rate = harmful_count / total_examples if total_examples else 0.0

    per_example_results: List[Dict[str, Any]] = []
    for record, judgement, language_info in zip(records, judgements, language_details):
        per_example_results.append(
            {
                "prompt": record.get("prompt"),
                "response": record.get("response"),
                "language": record.get("language"),
                "source": record.get("source"),
                "metadata": record.get("metadata"),
                "harmful": judgement.is_harmful,
                "confidence": judgement.confidence,
                "reasoning": judgement.reasoning,
                "response_language_label": language_info["predicted_label"],
                "response_language_score": language_info["predicted_score"],
                "expected_language_label": language_info["expected_label"],
                "language_match": language_info["language_match"],
            }
        )

    aggregate_results = {
        "total_examples": total_examples,
        "harmful_examples": harmful_count,
        "harmful_rate": harmful_rate,
        "per_example_results": per_example_results,
    }
    aggregate_results.update(language_summary)

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with args.results_path.open("w", encoding="utf-8") as fout:
        json.dump(aggregate_results, fout, ensure_ascii=False, indent=2)

    print(f"Evaluated {total_examples} examples.")
    print(f"Harmful rate: {harmful_rate:.3%} ({harmful_count}/{total_examples})")
    fidelity = aggregate_results["language_fidelity"]
    print(f"Language fidelity: {fidelity:.3%} ({language_summary['language_match_count']}/{language_summary['language_evaluable_examples']})")


if __name__ == "__main__":
    main()
