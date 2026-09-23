#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/subject.env}"
test -f "$CONFIG" || { echo "Missing config: $CONFIG" >&2; exit 2; }
set -a
source "$CONFIG"
set +a

if [[ "${EXECUTE:-0}" != "1" ]]; then
  echo "Dry run: set EXECUTE=1 to generate the 200 subject samples."
  echo "python code/lora/generate_subject_samples_200.py --model '$FLUX_MODEL' --lora '$LORA_WEIGHTS' --output '$SUBJECT_SAMPLE_DIR' --token '$CONCEPT_TOKEN' --subject '$SUBJECT' --seed-base '${SEED:-0}'"
  exit 0
fi

"${PYTHON_BIN:-python}" code/lora/generate_subject_samples_200.py \
  --model "$FLUX_MODEL" \
  --lora "$LORA_WEIGHTS" \
  --output "$SUBJECT_SAMPLE_DIR" \
  --token "$CONCEPT_TOKEN" \
  --subject "$SUBJECT" \
  --seed-base "${SEED:-0}"
