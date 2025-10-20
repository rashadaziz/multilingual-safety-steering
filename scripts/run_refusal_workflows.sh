#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-}"

if [[ -z "${MODE}" ]]; then
  cat <<'EOF'
Usage: scripts/run_refusal_workflows.sh MODE LANGUAGE_SPEC [EXTRA_ARGS...]

MODE must be one of baseline, refusal, refusal+ablate.
LANGUAGE_SPEC can be a single language (e.g., english), a comma-separated list
  (e.g., english,spanish,french), or the keyword 'all' to cover every supported language.
Any EXTRA_ARGS are passed through unchanged.
EOF
  exit 1
fi

shift

LANGUAGE_SPEC="${1:-}"
if [[ -z "${LANGUAGE_SPEC}" ]]; then
  echo "Missing LANGUAGE specification." >&2
  exit 1
fi

shift

ALL_LANGUAGES=(
  arabic
  chinese
  english
  french
  german
  italian
  japanese
  korean
  portuguese
  russian
  spanish
  thai
  vietnamese
)

if [[ "${LANGUAGE_SPEC}" == "all" ]]; then
  LANGUAGES=("${ALL_LANGUAGES[@]}")
else
  IFS=',' read -r -a LANGUAGES <<< "${LANGUAGE_SPEC}"
fi

FILTERED_LANGUAGES=()
for language in "${LANGUAGES[@]}"; do
  trimmed="${language//[[:space:]]/}"
  if [[ -n "${trimmed}" ]]; then
    # Normalize to lowercase to match enum values expected by the workflow.
    FILTERED_LANGUAGES+=("${trimmed,,}")
  fi
done

if [[ ${#FILTERED_LANGUAGES[@]} -eq 0 ]]; then
  echo "No valid languages provided in '${LANGUAGE_SPEC}'." >&2
  exit 1
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

OUTPUT_DIR="output/language-refusal"
mkdir -p "${OUTPUT_DIR}"

LAYERS=(14 15 16 17 18 19 20 21 22)
MODE_ARGS=()

case "${MODE}" in
  baseline)
    SUFFIX="_baseline"
    MODE_ARGS=(--refusal-strength 0)
    ;;
  refusal)
    SUFFIX="_refusal"
    MODE_ARGS=(--refusal-strength 2 --layers "${LAYERS[@]}")
    ;;
  refusal+ablate)
    SUFFIX="_refusal+ablate"
    MODE_ARGS=(--refusal-strength 2 --layers "${LAYERS[@]}" --language-means-path output/linguistic-subspace/MdL.pt)
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac

OUTPUT_PATH="${OUTPUT_DIR}/prompt_responses${SUFFIX}.json"
HARMLESS_OUTPUT_PATH="${OUTPUT_DIR}/prompt_responses_harmless${SUFFIX}.json"

CMD=(
  python -m workflows.refusal_vector_intervention
  --batch-size 32
  --output-path "${OUTPUT_PATH}"
  --harmless-output-path "${HARMLESS_OUTPUT_PATH}"
)

if [[ ${#FILTERED_LANGUAGES[@]} -eq 1 ]]; then
  CMD+=(--language "${FILTERED_LANGUAGES[0]}")
else
  CMD+=(--languages "${FILTERED_LANGUAGES[@]}")
fi
CMD+=("${MODE_ARGS[@]}")

echo "Running: ${CMD[*]} $*"
exec "${CMD[@]}" "$@"
