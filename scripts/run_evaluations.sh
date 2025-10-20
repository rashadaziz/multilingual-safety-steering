#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-}"

if [[ -z "${MODE}" ]]; then
  cat <<'EOF'
Usage: scripts/run_evaluations.sh MODE

MODE must be one of baseline, refusal, refusal+ablate.
Runs safety_evaluation and utility_evaluation with matching input/output suffixes.
EOF
  exit 1
fi

shift

if [[ $# -gt 0 ]]; then
  echo "Ignoring extra arguments: $*" >&2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

OUTPUT_DIR="output/language-refusal"
mkdir -p "${OUTPUT_DIR}"

case "${MODE}" in
  baseline)
    SUFFIX="_baseline"
    ;;
  refusal)
    SUFFIX="_refusal"
    ;;
  refusal+ablate)
    SUFFIX="_refusal+ablate"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac

SAFETY_INPUT="${OUTPUT_DIR}/prompt_responses${SUFFIX}.json"
UTILITY_INPUT="${OUTPUT_DIR}/prompt_responses_harmless${SUFFIX}.json"

SAFETY_RESULTS="${OUTPUT_DIR}/evaluation_summary${SUFFIX}.json"
SAFETY_DETAILS="${OUTPUT_DIR}/evaluation_details${SUFFIX}.json"

UTILITY_RESULTS="${OUTPUT_DIR}/utility_summary${SUFFIX}.json"
UTILITY_DETAILS="${OUTPUT_DIR}/utility_details${SUFFIX}.json"

echo "Running safety evaluation (${MODE})..."
python -m workflows.safety_evaluation \
  --input-path "${SAFETY_INPUT}" \
  --results-path "${SAFETY_RESULTS}" \
  --details-path "${SAFETY_DETAILS}" &
SAFETY_PID=$!

echo "Running utility evaluation (${MODE})..."
python -m workflows.utility_evaluation \
  --input-path "${UTILITY_INPUT}" \
  --results-path "${UTILITY_RESULTS}" \
  --details-path "${UTILITY_DETAILS}" &
UTILITY_PID=$!

FAIL=0

wait "${SAFETY_PID}" || FAIL=1
wait "${UTILITY_PID}" || FAIL=1

if [[ "${FAIL}" -ne 0 ]]; then
  echo "One or more evaluation jobs failed." >&2
  exit 1
fi

echo "Evaluations completed."
