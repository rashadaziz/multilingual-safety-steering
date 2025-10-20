#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-}"

if [[ -z "${MODE}" ]]; then
  cat <<'EOF'
Usage: scripts/run_refusal_workflows.sh MODE LANGUAGE [EXTRA_ARGS...]

MODE must be one of baseline, refusal, refusal+ablate.
LANGUAGE is forwarded to workflows.refusal_vector_intervention (e.g., english, spanish).
Any EXTRA_ARGS are passed through unchanged.
EOF
  exit 1
fi

shift

LANGUAGE="${1:-}"
if [[ -z "${LANGUAGE}" ]]; then
  echo "Missing LANGUAGE argument." >&2
  exit 1
fi

shift

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
  --language "${LANGUAGE}"
  --output-path "${OUTPUT_PATH}"
  --harmless-output-path "${HARMLESS_OUTPUT_PATH}"
)
CMD+=("${MODE_ARGS[@]}")

echo "Running: ${CMD[*]} $*"
exec "${CMD[@]}" "$@"
