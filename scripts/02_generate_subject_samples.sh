#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/subject.env}"
test -f "$CONFIG" || { echo "Missing config: $CONFIG" >&2; exit 2; }
set -a
source "$CONFIG"
set +a

EXECUTE_FLAG=()
[[ "${EXECUTE:-0}" == "1" ]] && EXECUTE_FLAG=(--execute)

"${PYTHON_BIN:-python}" training/generate_subject_samples.py \
  --model-id "$FLUX_MODEL" \
  --lora-path "$LORA_WEIGHTS" \
  --prompt "$SAMPLE_PROMPT" \
  --output-dir "$SUBJECT_SAMPLE_DIR" \
  --num-images 200 \
  --seed "${SEED:-0}" \
  "${EXECUTE_FLAG[@]}"
