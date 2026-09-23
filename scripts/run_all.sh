#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/subject.env}"
bash scripts/01_train_lora.sh "$CONFIG"
bash scripts/02_generate_subject_samples.sh "$CONFIG"
echo "Select one clean, single-subject sample and set SOURCE_IMAGE in $CONFIG."
echo "Then run: bash scripts/03_run_carrier_attacks.sh $CONFIG"
