#!/usr/bin/env bash
set -euo pipefail

CONVERSATION="${1:-0}"

python -m memmark.examples.run_locomo_full \
  --locomo "${MEMMARK_LOCOMO_PATH:?Set MEMMARK_LOCOMO_PATH}" \
  --conversation "$CONVERSATION" \
  --backend json \
  --llm-mode stub \
  --max-sessions "${MAX_SESSIONS:-2}" \
  --max-qa "${MAX_QA:-5}" \
  --baselines watermark no_watermark \
  --output-mode metrics
