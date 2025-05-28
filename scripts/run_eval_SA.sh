#!/bin/bash
YEAR=$1
DOY=$2
CONFIG_PATH=${3:-config/config.yaml}

# One single Python call does GNSS, Fusion-SW, Fusion-LW, DTEC-Fusion-LW and DTEC-Fusion-SW in sequence:
python src/eval_with_SA.py --year "$YEAR" --doy "$DOY" --config "$CONFIG_PATH" # --force use force to overwrite the SA metrics

echo "All evaluations have completed."
