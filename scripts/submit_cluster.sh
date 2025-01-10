#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1024
#SBATCH --gpus=1

# config over #
%"#SBATCH --output=/dev/null"
%"#SBATCH --output=output_files/output.out"


module load stack/2024-06 python_cuda/3.11.6
module load eth_proxy

main_dir="/cluster/work/igp_psr/arrueegg/GIM_fusion_VLBI/"
year=$1
doy=$2

source ${main_dir}/env/bin/activate
bash ${main_dir}/scripts/run_dtec.sh $year $doy