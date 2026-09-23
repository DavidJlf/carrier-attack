#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/subject.env}"
test -f "$CONFIG" || { echo "Missing config: $CONFIG" >&2; exit 2; }
set -a
source "$CONFIG"
set +a

EXECUTE_FLAG=()
[[ "${EXECUTE:-0}" == "1" ]] && EXECUTE_FLAG=(--execute)

"${PYTHON_BIN:-python}" code/lora/train_flux_lora.py \
  --trainer "$FLUX_LORA_TRAINER" \
  --accelerate "${ACCELERATE_BIN:-accelerate}" \
  --model_id "$FLUX_MODEL" \
  --instance_data_dir "$REFERENCE_DIR" \
  --instance_prompt "$INSTANCE_PROMPT" \
  --output_dir "$LORA_OUTPUT_DIR" \
  --validation_prompt "$SAMPLE_PROMPT" \
  "${EXECUTE_FLAG[@]}"
