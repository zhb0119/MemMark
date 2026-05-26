#!/usr/bin/env bash
set -euo pipefail

slugify() {
  local value="${1:-model}"
  value="$(printf '%s' "$value" | sed -E 's/[^A-Za-z0-9._+-]+/-/g; s/^-+//; s/-+$//')"
  if [[ -z "$value" ]]; then
    value="model"
  fi
  printf '%s' "$value"
}

resolve_model_name() {
  if [[ -n "${RESULT_MODEL_NAME:-}" ]]; then
    printf '%s' "$RESULT_MODEL_NAME"
  elif [[ -n "${TARGET_LLM_MODEL:-}" ]]; then
    printf '%s' "$TARGET_LLM_MODEL"
  elif [[ -n "${MEMMARK_MODEL:-}" ]]; then
    printf '%s' "$MEMMARK_MODEL"
  elif [[ -n "${OPENAI_MODEL:-}" ]]; then
    printf '%s' "$OPENAI_MODEL"
  else
    printf '%s' "model"
  fi
}

resolve_run_tag() {
  if [[ -n "${RUN_TAG:-}" ]]; then
    printf '%s' "$RUN_TAG"
  elif [[ -n "${MEMMARK_RUN_TAG:-}" ]]; then
    printf '%s' "$MEMMARK_RUN_TAG"
  else
    date +%Y%m%d-%H%M%S
  fi
}

make_output_file() {
  local memory_system="$1"
  local conversation="$2"
  local baseline="$3"
  local output_root="${OUTPUT_ROOT:-results}"
  local model_slug
  local run_tag
  model_slug="$(slugify "$(resolve_model_name)")"
  run_tag="$(slugify "$(resolve_run_tag)")"
  printf '%s/%s/%s/%s/conv%s_%s.json' \
    "$output_root" "$memory_system" "$model_slug" "$run_tag" "$conversation" "$baseline"
}

# Usage:
#   set -a; source .env; set +a
#   bash scripts/run_locomo_amem.sh 0 watermark

CONVERSATION="${1:-0}"
BASELINE="${2:-watermark}"
if [[ -z "${OUTPUT_FILE:-}" ]]; then
  OUTPUT_FILE="$(make_output_file amem "$CONVERSATION" "$BASELINE")"
fi
AMEM_MODEL_NAME="${AMEM_MODEL_NAME:-sentence-transformers/all-MiniLM-L6-v2}"
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "Output: $OUTPUT_FILE"

# A-MEM uses OPENAI_* internally. Keep this mapping local to the A-MEM script
# so the same .env can also carry Graphiti's internal OpenAI-compatible config.
export OPENAI_BASE_URL="${AMEM_OPENAI_BASE_URL:-${TARGET_LLM_BASE:-${MEMMARK_BASE_URL:-}}}"
export OPENAI_API_KEY="${AMEM_OPENAI_API_KEY:-${TARGET_LLM_API_KEY:-${MEMMARK_API_KEY:-}}}"
export OPENAI_MODEL="${AMEM_OPENAI_MODEL:-${TARGET_LLM_MODEL:-${MEMMARK_MODEL:-}}}"
if [[ -n "${AMEM_OPENAI_EXTRA_BODY:-}" ]]; then
  export OPENAI_EXTRA_BODY="$AMEM_OPENAI_EXTRA_BODY"
fi

export MEMMARK_BASE_URL="${MEMMARK_BASE_URL:-${TARGET_LLM_BASE:-$OPENAI_BASE_URL}}"
export MEMMARK_API_KEY="${MEMMARK_API_KEY:-${TARGET_LLM_API_KEY:-$OPENAI_API_KEY}}"
export MEMMARK_MODEL="${MEMMARK_MODEL:-${TARGET_LLM_MODEL:-$OPENAI_MODEL}}"
export TARGET_LLM_BASE="${TARGET_LLM_BASE:-$MEMMARK_BASE_URL}"
export TARGET_LLM_API_KEY="${TARGET_LLM_API_KEY:-$MEMMARK_API_KEY}"
export TARGET_LLM_MODEL="${TARGET_LLM_MODEL:-$MEMMARK_MODEL}"

python -m memmark.examples.run_locomo_full \
  --locomo "${MEMMARK_LOCOMO_PATH:?Set MEMMARK_LOCOMO_PATH}" \
  --conversation "$CONVERSATION" \
  --backend amem \
  --amem-model-name "$AMEM_MODEL_NAME" \
  --llm-mode real \
  --progress \
  --async-assess \
  --async-max-concurrency "${ASYNC_MAX_CONCURRENCY:-2}" \
  --max-sessions "${MAX_SESSIONS:-1}" \
  --max-qa "${MAX_QA:-10}" \
  --baselines "$BASELINE" \
  --output-mode metrics \
  --output "$OUTPUT_FILE"
