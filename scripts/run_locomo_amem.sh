#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   set -a; source .env; set +a
#   bash scripts/run_locomo_amem.sh 0 watermark

CONVERSATION="${1:-0}"
BASELINE="${2:-watermark}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
MODEL_NAME="${RESULT_MODEL_NAME:-${TARGET_LLM_MODEL:-${MEMMARK_MODEL:-${OPENAI_MODEL:-model}}}}"
MODEL_SLUG="$(printf '%s' "$MODEL_NAME" | tr -cs '[:alnum:]._-+' '-')"
MODEL_SLUG="${MODEL_SLUG#-}"; MODEL_SLUG="${MODEL_SLUG%-}"
RUN_TAG="${RUN_TAG:-${MEMMARK_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_ROOT/amem/${MODEL_SLUG:-model}/$RUN_TAG/conv${CONVERSATION}_${BASELINE}.json}"
AMEM_MODEL_NAME="${AMEM_MODEL_NAME:-sentence-transformers/all-MiniLM-L6-v2}"
mkdir -p "$(dirname "$OUTPUT_FILE")"

# A-MEM uses OPENAI_* internally. Keep this mapping local to the A-MEM script
# so the same .env can also carry Graphiti's internal OPENAI-compatible config.
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
