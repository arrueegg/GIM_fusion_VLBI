#!/bin/bash

# assign the first argument to the $YEAR variable
YEAR=$1
# assign the second argument to the $DOY variable
DOY=$2
# assign the third argument to the $CONFIG_PATH variable if provided, otherwise use default config/config.yaml
CONFIG_PATH=${3:-config/config.yaml}

#cd /scratch2/arrueegg/WP2/GIM_fusion_VLBI/
#source env/bin/activate

LOSS_FN="LaplaceLoss"
EXPERIMENTS_DIR="/cluster/work/igp_psr/arrueegg/WP2/GIM_fusion_VLBI"

TARGET_DIR="${EXPERIMENTS_DIR}/${MODE}_${YEAR}_${DOY}_SW1_LW1/SA_plots"
METRICS_FILE="${TARGET_DIR}/metrics.txt"
if [ ! -f "$METRICS_FILE" ]; then
    echo "Running GNSS with LaplaceLoss"
    python src/train.py --debug False --config_path "$CONFIG_PATH" --year "$YEAR" --doy "$DOY" --mode GNSS --loss_fn "$LOSS_FN"
    python src/inference_map.py --debug False --config_path "$CONFIG_PATH" --year "$YEAR" --doy "$DOY" --mode GNSS --loss_fn "$LOSS_FN"
    python src/eval_with_SA.py --debug False --config_path "$CONFIG_PATH" --year "$YEAR" --doy "$DOY" --mode GNSS --loss_fn "$LOSS_FN"

    wandb sync wandb/
else
    echo "Skipping execution for $YEAR $DOY GNSS; as metrics file already exists."
fi

echo "Running Fusion with LaplaceLoss and vlbi_sampling_weight 1000.0"
python src/train.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 1000.0
python src/inference_map.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 1000.0
python src/eval_with_SA.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 1000.0
wandb sync wandb/

echo "Running Fusion with LaplaceLoss and vlbi_loss_weight 1000.0"
python src/train.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 1000.0
python src/inference_map.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 1000.0
python src/eval_with_SA.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 1000.0
wandb sync wandb/

echo "Running DTEC Fusion with LaplaceLoss and vlbi_loss_weight 100.0"
python src/train.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode DTEC_Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 100.0
python src/inference_map.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode DTEC_Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 100.0
python src/eval_with_SA.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode DTEC_Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 100.0
wandb sync wandb/

echo "Running DTEC Fusion with LaplaceLoss and vlbi_sampling_weight 100.0"
python src/train.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode DTEC_Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 100.0
python src/inference_map.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode DTEC_Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 100.0
python src/eval_with_SA.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode DTEC_Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 100.0
wandb sync wandb/

# Wait for all background processes to finish
wait

echo "All training instances have completed."
