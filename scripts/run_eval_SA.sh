#!/bin/bash

# Assign arguments to variables
YEAR=$1
DOY=$2
file_doy=$(printf "%03d" $DOY)
CONFIG_PATH=${3:-config/config.yaml}

LOSS_FN="LaplaceLoss"
EXPERIMENTS_DIR="/cluster/work/igp_psr/arrueegg/WP2/GIM_fusion_VLBI/experiments"

# Run evaluation for GNSS
TARGET_DIR="${EXPERIMENTS_DIR}/GNSS_${YEAR}_${file_doy}_SW1_LW1/SA_plots"
METRICS_FILE="${TARGET_DIR}/metrics.txt"
if [ ! -f "$METRICS_FILE" ]; then
    echo "Evaluating GNSS with LaplaceLoss"
    python src/eval_with_SA.py --debug False --config_path "$CONFIG_PATH" --year "$YEAR" --doy "$DOY" --mode "GNSS" --loss_fn "$LOSS_FN"
fi

# Run evaluation for Fusion with vlbi_sampling_weight 1000.0
TARGET_DIR="${EXPERIMENTS_DIR}/Fusion_${YEAR}_${file_doy}_SW1000_LW1/SA_plots"
METRICS_FILE="${TARGET_DIR}/metrics.txt"
if [ ! -f "$METRICS_FILE" ]; then
    echo "Evaluating Fusion with LaplaceLoss and vlbi_sampling_weight 1000.0"
    python src/eval_with_SA.py --debug False --config_path "$CONFIG_PATH" --year "$YEAR" --doy "$DOY" --mode "Fusion" --loss_fn "$LOSS_FN" --vlbi_sampling_weight 1000.0
fi

# Run evaluation for Fusion with vlbi_loss_weight 1000.0
TARGET_DIR="${EXPERIMENTS_DIR}/Fusion_${YEAR}_${file_doy}_SW1_LW1000/SA_plots"
METRICS_FILE="${TARGET_DIR}/metrics.txt"
if [ ! -f "$METRICS_FILE" ]; then
    echo "Evaluating Fusion with LaplaceLoss and vlbi_loss_weight 1000.0"
    python src/eval_with_SA.py --debug False --config_path "$CONFIG_PATH" --year "$YEAR" --doy "$DOY" --mode "Fusion" --loss_fn "$LOSS_FN" --vlbi_loss_weight 1000.0
fi

# Run evaluation for DTEC Fusion with vlbi_loss_weight 100.0
TARGET_DIR="${EXPERIMENTS_DIR}/DTEC_Fusion_${YEAR}_${file_doy}_SW1_LW100/SA_plots"
METRICS_FILE="${TARGET_DIR}/metrics.txt"
if [ ! -f "$METRICS_FILE" ]; then
    echo "Evaluating DTEC Fusion with LaplaceLoss and vlbi_loss_weight 100.0"
    python src/eval_with_SA.py --debug False --config_path "$CONFIG_PATH" --year "$YEAR" --doy "$DOY" --mode "DTEC_Fusion" --loss_fn "$LOSS_FN" --vlbi_loss_weight 100.0
fi

# Run evaluation for DTEC Fusion with vlbi_sampling_weight 100.0
TARGET_DIR="${EXPERIMENTS_DIR}/DTEC_Fusion_${YEAR}_${file_doy}_SW100_LW1/SA_plots"
METRICS_FILE="${TARGET_DIR}/metrics.txt"
if [ ! -f "$METRICS_FILE" ]; then
    echo "Evaluating DTEC Fusion with LaplaceLoss and vlbi_sampling_weight 100.0"
    python src/eval_with_SA.py --debug False --config_path "$CONFIG_PATH" --year "$YEAR" --doy "$DOY" --mode "DTEC_Fusion" --loss_fn "$LOSS_FN" --vlbi_sampling_weight 100.0
fi

# Wait for all background processes to finish
wait
echo "All evaluations have completed."