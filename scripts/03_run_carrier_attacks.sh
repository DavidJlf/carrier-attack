#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/subject.env}"
test -f "$CONFIG" || { echo "Missing config: $CONFIG" >&2; exit 2; }
set -a
source "$CONFIG"
set +a

if [[ "${VISIBLE_CARRIER:-auto}" == "auto" ]]; then
  read -r VISIBLE_CARRIER_CLASS VISIBLE_CARRIER < <(
    "${PYTHON_BIN:-python}" code/formal_pipeline/carrier_catalog.py --target-class "$TARGET_CLASS"
  )
elif [[ "${VISIBLE_CARRIER_CLASS:-auto}" == "auto" ]]; then
  echo "Set VISIBLE_CARRIER_CLASS when using a custom VISIBLE_CARRIER." >&2
  exit 2
fi

COMMON=(
  --image "$SOURCE_IMAGE"
  --subject "$SUBJECT"
  --source-case "$SUBJECT"
  --concept-token "$CONCEPT_TOKEN"
  --source-classes "$SOURCE_CLASSES"
  --lora-path "$LORA_WEIGHTS"
  --target "$TARGET_LABEL"
  --target-class "$TARGET_CLASS"
  --visible-carrier "$VISIBLE_CARRIER"
  --methods "${METHODS:-cra,jia,cira}"
  --seed "${SEED:-0}"
  --output-root "$OUTPUT_ROOT"
  --run-name "$RUN_NAME"
  --sam-python "${SAM_PYTHON:-${PYTHON_BIN:-python}}"
)

if [[ "${EXECUTE:-0}" != "1" ]]; then
  "${PYTHON_BIN:-python}" code/formal_pipeline/flux_automation.py "${COMMON[@]}" --run-name "${RUN_NAME}_dry_run_$$" --dry-run
  echo "Execution also runs the carrier quality gate and optional DINOv3/SAM3 preservation evaluation."
  exit 0
fi

RUN_ROOT="$OUTPUT_ROOT/$RUN_NAME"
"${PYTHON_BIN:-python}" code/formal_pipeline/flux_automation.py "${COMMON[@]}" --phase prepare
"${PYTHON_BIN:-python}" code/formal_pipeline/run_carrier_composite_gate.py \
  --run-root "$RUN_ROOT" \
  --carrier-label "$VISIBLE_CARRIER" \
  --carrier-class "$VISIBLE_CARRIER_CLASS"
"${PYTHON_BIN:-python}" code/formal_pipeline/flux_automation.py "${COMMON[@]}" --phase attack

if [[ -n "${DINO_MODEL:-}" ]]; then
  "${PYTHON_BIN:-python}" code/formal_pipeline/evaluate_subject_preservation.py \
    --run-root "$RUN_ROOT" --model "$DINO_MODEL" \
    --sam3-python "${SAM_PYTHON:-${PYTHON_BIN:-python}}" \
    --sam3-checkpoint "$SAM3_CHECKPOINT" --execute
fi
