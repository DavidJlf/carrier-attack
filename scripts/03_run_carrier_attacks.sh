#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/subject.env}"
test -f "$CONFIG" || { echo "Missing config: $CONFIG" >&2; exit 2; }
set -a
source "$CONFIG"
set +a

COMMON=(
  --image "$SOURCE_IMAGE"
  --subject "$SUBJECT"
  --source-case "$SUBJECT"
  --concept-token "$CONCEPT_TOKEN"
  --source-classes "$SOURCE_CLASSES"
  --lora-path "$LORA_WEIGHTS"
  --target "$TARGET_LABEL"
  --target-class "$TARGET_CLASS"
  --visible-anchor "$VISIBLE_CARRIER"
  --methods "${METHODS:-cra,jia,cira}"
  --seed "${SEED:-0}"
  --output-root "$OUTPUT_ROOT"
  --run-name "$RUN_NAME"
  --sam-python "${SAM_PYTHON:-${PYTHON_BIN:-python}}"
)

if [[ "${EXECUTE:-0}" != "1" ]]; then
  "${PYTHON_BIN:-python}" code/formal_pipeline/flux_auomation.py "${COMMON[@]}" --run-name "${RUN_NAME}_dry_run_$$" --dry-run
  echo "Execution also runs the carrier quality gate and optional DINOv3/SAM3 preservation evaluation."
  exit 0
fi

RUN_ROOT="$OUTPUT_ROOT/$RUN_NAME"
"${PYTHON_BIN:-python}" code/formal_pipeline/flux_auomation.py "${COMMON[@]}" --phase prepare
"${PYTHON_BIN:-python}" code/formal_pipeline/run_anchor_composite_gate.py \
  --run-root "$RUN_ROOT" \
  --anchor-label "$VISIBLE_CARRIER" \
  --anchor-class "${VISIBLE_CARRIER_CLASS:-$TARGET_CLASS}"
"${PYTHON_BIN:-python}" code/formal_pipeline/flux_auomation.py "${COMMON[@]}" --phase attack

if [[ -n "${DINO_MODEL:-}" ]]; then
  "${PYTHON_BIN:-python}" code/formal_pipeline/evaluate_subject_preservation_batch.py \
    --run-root "$RUN_ROOT" --model "$DINO_MODEL" \
    --sam3-python "${SAM_PYTHON:-${PYTHON_BIN:-python}}" \
    --sam3-checkpoint "$SAM3_CHECKPOINT" --execute
fi
