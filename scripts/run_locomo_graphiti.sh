#!/usr/bin/env bash
set -euo pipefail

# Start Neo4j first:
#   docker compose -f docker-compose.neo4j.yml up -d
# Then source .env and run:
#   set -a; source .env; set +a
#   bash scripts/run_locomo_graphiti.sh 0 watermark

CONVERSATION="${1:-0}"
BASELINE="${2:-watermark}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
MODEL_NAME="${RESULT_MODEL_NAME:-${TARGET_LLM_MODEL:-${MEMMARK_MODEL:-${OPENAI_MODEL:-model}}}}"
MODEL_SLUG="$(printf '%s' "$MODEL_NAME" | tr -cs '[:alnum:]._-+' '-')"
MODEL_SLUG="${MODEL_SLUG#-}"; MODEL_SLUG="${MODEL_SLUG%-}"
RUN_TAG="${RUN_TAG:-${MEMMARK_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_ROOT/graphiti/${MODEL_SLUG:-model}/$RUN_TAG/conv${CONVERSATION}_${BASELINE}.json}"
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Graphiti uses OPENAI_* for entity/relation extraction. Keep this mapping local
# to the Graphiti script so A-MEM can use its own native LLM config.
export OPENAI_BASE_URL="${GRAPHITI_OPENAI_BASE_URL:-${OPENAI_BASE_URL:-${MEMMARK_BASE_URL:-}}}"
export OPENAI_API_KEY="${GRAPHITI_OPENAI_API_KEY:-${OPENAI_API_KEY:-${MEMMARK_API_KEY:-}}}"
export OPENAI_MODEL="${GRAPHITI_OPENAI_MODEL:-${OPENAI_MODEL:-${MEMMARK_MODEL:-}}}"

export GRAPHITI_RERANKER_BASE_URL="${GRAPHITI_RERANKER_BASE_URL:-$OPENAI_BASE_URL}"
export GRAPHITI_RERANKER_API_KEY="${GRAPHITI_RERANKER_API_KEY:-$OPENAI_API_KEY}"
export GRAPHITI_RERANKER_MODEL="${GRAPHITI_RERANKER_MODEL:-$OPENAI_MODEL}"

export TARGET_LLM_BASE="${TARGET_LLM_BASE:-${MEMMARK_BASE_URL:-$OPENAI_BASE_URL}}"
export TARGET_LLM_API_KEY="${TARGET_LLM_API_KEY:-${MEMMARK_API_KEY:-$OPENAI_API_KEY}}"
export TARGET_LLM_MODEL="${TARGET_LLM_MODEL:-${MEMMARK_MODEL:-$OPENAI_MODEL}}"
export MEMMARK_BASE_URL="${MEMMARK_BASE_URL:-$TARGET_LLM_BASE}"
export MEMMARK_API_KEY="${MEMMARK_API_KEY:-$TARGET_LLM_API_KEY}"
export MEMMARK_MODEL="${MEMMARK_MODEL:-$TARGET_LLM_MODEL}"

python -m memmark.examples.run_locomo_full \
  --locomo "${MEMMARK_LOCOMO_PATH:?Set MEMMARK_LOCOMO_PATH}" \
  --conversation "$CONVERSATION" \
  --backend graphiti \
  --llm-mode real \
  --progress \
  --async-assess \
  --async-max-concurrency "${ASYNC_MAX_CONCURRENCY:-2}" \
  --max-sessions "${MAX_SESSIONS:-999}" \
  --max-qa "${MAX_QA:-999}" \
  --baselines "$BASELINE" \
  --output-mode metrics \
  --output "$OUTPUT_FILE"
