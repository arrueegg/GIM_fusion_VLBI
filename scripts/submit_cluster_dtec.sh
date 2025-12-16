#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/cluster/work/igp_psr/arrueegg/WP2/logs/slurm_DTEC-%j.out

module load stack/2024-06 python_cuda/3.11.6
module load eth_proxy

main_dir="/cluster/work/igp_psr/arrueegg/WP2/GIM_fusion_VLBI/"
year=$1
doy=$2
config_path="config/config_cluster.yaml"

cd $main_dir
source ${main_dir}/env/bin/activate

file_doy=$(printf "%03d" $doy)
LOSS_FN="LaplaceLoss"
EXPERIMENTS_DIR="${main_dir}/experiments"

echo "================================================================================"
echo "Training DTEC model for ${year}_${file_doy}"
echo "================================================================================"

# Train DTEC model (DTEC_Fusion SW1_LW100)
TARGET_DIR="${EXPERIMENTS_DIR}/DTEC_Fusion_${year}_${file_doy}_SW1_LW100/model"
if [ ! -d "$TARGET_DIR" ] || [ -z "$(ls -A $TARGET_DIR)" ]; then
    echo "Training DTEC Fusion with vlbi_loss_weight 100.0"
    python src/train.py --debug False --config_path "$config_path" --year "$year" --doy "$doy" --mode "DTEC_Fusion" --loss_fn "$LOSS_FN" --vlbi_loss_weight 100.0
    python src/inference_map.py --debug False --config_path "$config_path" --year "$year" --doy "$doy" --mode "DTEC_Fusion" --loss_fn "$LOSS_FN" --vlbi_loss_weight 100.0
    python src/eval_with_SA.py --year "$year" --doy "$doy" --config "$config_path"
    wandb sync wandb/
else
    echo "DTEC model already exists, skipping"
fi

echo "================================================================================"
echo "Completed training for ${year}_${file_doy}"
echo "================================================================================"
