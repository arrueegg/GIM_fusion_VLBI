#!/bin/bash

cd /scratch2/arrueegg/WP2/GIM_fusion_VLBI/

python src/train.py --debug False --year 2023 --doy 010 --mode GNSS --loss_fn LaplaceLoss
python src/inference_map.py --debug False --year 2023 --doy 010 --mode GNSS --loss_fn LaplaceLoss

python src/train.py --debug False --year 2023 --doy 010 --mode GNSS --loss_fn MSELoss
python src/inference_map.py --debug False --year 2023 --doy 010 --mode GNSS --loss_fn MSELoss

python src/train.py --debug False --year 2023 --doy 010 --mode GNSS --loss_fn GaussianNLLLoss
python src/inference_map.py --debug False --year 2023 --doy 010 --mode GNSS --loss_fn GaussianNLLLoss

python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss

python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 10.0
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 10.0

python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 100.0
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 100.0

python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 1000.0
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 1000.0

python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 10000.0
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 10000.0

python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 10.0
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 10.0

python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 100.0
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 100.0

python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 1000.0
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 1000.0

python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 10000.0
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 10000.0


# Wait for all background processes to finish
wait

echo "All training instances have completed."
