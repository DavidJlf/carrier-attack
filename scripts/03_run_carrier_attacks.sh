#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/subject.env}"
test -f "$CONFIG" || { echo "Missing config: $CONFIG" >&2; exit 2; }
set -a
source "$CONFIG"
set +a

CONDITION="${CONSTRUCTION_CONDITION:-non_target_carrier}"
CASE_NAME="${RUN_NAME}_${CONDITION}"
COMMON=(
  --image "$SOURCE_IMAGE"
  --subject "$SUBJECT"
  --source-case "$SUBJECT"
  --concept-token "$CONCEPT_TOKEN"
  --source-classes "$SOURCE_CLASSES"
  --lora-path "$LORA_WEIGHTS"
  --target "$TARGET_LABEL"
  --target-class "$TARGET_CLASS"
  --construction-condition "$CONDITION"
  --hybrid-feature-clause "${HYBRID_FEATURE_CLAUSE:-}"
  --hybrid-identity-clause "${HYBRID_IDENTITY_CLAUSE:-}"
  --hybrid-forbidden-clause "${HYBRID_FORBIDDEN_CLAUSE:-}"
  --seed "${SEED:-0}"
  --output-root "$OUTPUT_ROOT"
  --run-name "$CASE_NAME"
  --sam-python "${SAM_PYTHON:-${PYTHON_BIN:-python}}"
)

if [[ "${EXECUTE:-0}" != "1" ]]; then
  "${PYTHON_BIN:-python}" code/formal_pipeline/flux_automation.py \
    "${COMMON[@]}" --run-name "${CASE_NAME}_dry_run_$$" --dry-run
  exit 0
fi

RUN_ROOT="$OUTPUT_ROOT/$CASE_NAME"
"${PYTHON_BIN:-python}" code/formal_pipeline/flux_automation.py "${COMMON[@]}" --phase prepare
CARRIER_INFO="$(
  "${PYTHON_BIN:-python}" -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); print(value["carrier_class"]); print(value["visible_carrier"])' \
    "$RUN_ROOT/manifest.json"
)"
mapfile -t CARRIER_SPEC <<< "$CARRIER_INFO"
"${PYTHON_BIN:-python}" code/formal_pipeline/run_carrier_composite_gate.py \
  --run-root "$RUN_ROOT" --carrier-label "${CARRIER_SPEC[1]}" --carrier-class "${CARRIER_SPEC[0]}"
"${PYTHON_BIN:-python}" code/formal_pipeline/flux_automation.py "${COMMON[@]}" --phase attack

if [[ -n "${DINO_MODEL:-}" ]]; then
  "${PYTHON_BIN:-python}" code/formal_pipeline/evaluate_subject_preservation.py \
    --run-root "$RUN_ROOT" --model "$DINO_MODEL" \
    --sam3-python "${SAM_PYTHON:-${PYTHON_BIN:-python}}" \
    --sam3-checkpoint "$SAM3_CHECKPOINT" --execute
fi
