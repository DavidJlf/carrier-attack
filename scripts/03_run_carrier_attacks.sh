#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/subject.env}"
test -f "$CONFIG" || { echo "Missing config: $CONFIG" >&2; exit 2; }
set -a
source "$CONFIG"
set +a

RUN_FLAG=(--dry-run)
[[ "${EXECUTE:-0}" == "1" ]] && RUN_FLAG=()

"${PYTHON_BIN:-python}" src/carrier/run_pipeline.py \
  --image "$SOURCE_IMAGE" \
  --subject "$SUBJECT" \
  --source-case "$SUBJECT" \
  --concept-token "$CONCEPT_TOKEN" \
  --source-classes "$SOURCE_CLASSES" \
  --lora-path "$LORA_WEIGHTS" \
  --target "$TARGET_LABEL" \
  --target-class "$TARGET_CLASS" \
  --visible-anchor "$VISIBLE_CARRIER" \
  --methods "${METHODS:-cra,jia,cira}" \
  --seed "${SEED:-0}" \
  --output-root "$OUTPUT_ROOT" \
  "${RUN_FLAG[@]}"
