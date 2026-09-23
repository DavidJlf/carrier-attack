#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/subject.env}"
bash scripts/01_train_lora.sh "$CONFIG"
bash scripts/02_generate_subject_samples.sh "$CONFIG"

set -a
source "$CONFIG"
set +a
if [[ -n "${SOURCE_IMAGE:-}" && -f "$SOURCE_IMAGE" ]]; then
  bash scripts/03_run_carrier_attacks.sh "$CONFIG"
else
  echo "Subject-sample candidates are ready."
  echo "Select one clean, single-subject sample, set SOURCE_IMAGE in $CONFIG, then rerun this script."
fi
