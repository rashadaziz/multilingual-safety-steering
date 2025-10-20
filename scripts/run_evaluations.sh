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

PIDS=()

terminate() {
  local signal="$1"
  local exit_code="$2"

  echo "Received ${signal}, terminating evaluation jobs..." >&2

  if [[ ${#PIDS[@]} -gt 0 ]]; then
    for pid in "${PIDS[@]}"; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill -TERM "${pid}" 2>/dev/null || true
      fi
    done

    sleep 1

    for pid in "${PIDS[@]}"; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill -KILL "${pid}" 2>/dev/null || true
      fi
    done

    for pid in "${PIDS[@]}"; do
      if [[ -n "${pid}" ]]; then
        wait "${pid}" 2>/dev/null || true
      fi
    done
  fi

  trap - INT TERM
  exit "${exit_code}"
}

trap 'terminate SIGINT 130' INT
trap 'terminate SIGTERM 143' TERM

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
UTILITY_RESULTS="${OUTPUT_DIR}/utility_summary${SUFFIX}.json"

echo "Running safety evaluation (${MODE})..."
python -m workflows.safety_evaluation \
  --model gpt-4o \
  --input-path "${SAFETY_INPUT}" \
  --results-path "${SAFETY_RESULTS}" &
SAFETY_PID=$!
PIDS+=("${SAFETY_PID}")

echo "Running utility evaluation (${MODE})..."
python -m workflows.utility_evaluation \
  --model gpt-4o \
  --input-path "${UTILITY_INPUT}" \
  --results-path "${UTILITY_RESULTS}" &
UTILITY_PID=$!
PIDS+=("${UTILITY_PID}")

FAIL=0

wait "${SAFETY_PID}" || FAIL=1
wait "${UTILITY_PID}" || FAIL=1

trap - INT TERM

if [[ "${FAIL}" -ne 0 ]]; then
  echo "One or more evaluation jobs failed." >&2
  exit 1
fi

echo "Evaluations completed."
