#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/subject.env}"
set -a
source "$CONFIG"
set +a
if [[ ! -f "$LORA_WEIGHTS" ]]; then
  bash scripts/01_train_lora.sh "$CONFIG"
fi
if [[ ! -f "$LORA_WEIGHTS" ]]; then
  echo "LoRA weights are not available yet. Finish training, then rerun this script."
  exit 0
fi
if [[ ! -f "$SUBJECT_SAMPLE_DIR/metadata.csv" ]]; then
  bash scripts/02_generate_subject_samples.sh "$CONFIG"
fi
if [[ ! -f "$SUBJECT_SAMPLE_DIR/metadata.csv" ]]; then
  echo "Subject-sample generation has not completed yet."
  exit 0
fi
if [[ -n "${SOURCE_IMAGE:-}" && -f "$SOURCE_IMAGE" ]]; then
  bash scripts/03_run_carrier_attacks.sh "$CONFIG"
else
  echo "Subject-sample candidates are ready."
  echo "Select one clean, single-subject sample, set SOURCE_IMAGE in $CONFIG, then rerun this script."
fi
