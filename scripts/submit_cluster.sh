#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --output=/cluster/work/igp_psr/arrueegg/WP2/logs/slurm_ML-%j.out

#export OMP_NUM_THREADS=24
#export OPENBLAS_NUM_THREADS=24

module load stack/2024-06 python_cuda/3.11.6
module load eth_proxy

main_dir="/cluster/work/igp_psr/arrueegg/WP2/GIM_fusion_VLBI/"
year=$1
doy=$2
config_path="config/config_cluster.yaml"

cd $main_dir
source ${main_dir}/env/bin/activate
bash ${main_dir}/scripts/run_dtec.sh $year $doy $config_path