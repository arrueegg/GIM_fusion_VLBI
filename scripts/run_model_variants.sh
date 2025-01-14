#!/bin/bash

cd /scratch2/arrueegg/WP2/GIM_fusion_VLBI/

echo "Running GNSS with LaplaceLoss"
python src/train.py --debug False --year 2023 --doy 010 --mode GNSS --loss_fn LaplaceLoss
python src/inference_map.py --debug False --year 2023 --doy 010 --mode GNSS --loss_fn LaplaceLoss

echo "Running GNSS with MSELoss"
python src/train.py --debug False --year 2023 --doy 010 --mode GNSS --loss_fn MSELoss
python src/inference_map.py --debug False --year 2023 --doy 010 --mode GNSS --loss_fn MSELoss

echo "Running GNSS with GaussianNLLLoss"
python src/train.py --debug False --year 2023 --doy 010 --mode GNSS --loss_fn GaussianNLLLoss
python src/inference_map.py --debug False --year 2023 --doy 010 --mode GNSS --loss_fn GaussianNLLLoss

echo "Running Fusion with LaplaceLoss"
python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss

echo "Running Fusion with LaplaceLoss and vlbi_loss_weight 10.0"
python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 10.0
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 10.0

echo "Running Fusion with LaplaceLoss and vlbi_loss_weight 100.0"
python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 100.0
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 100.0

echo "Running Fusion with LaplaceLoss and vlbi_loss_weight 1000.0"
python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 1000.0
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 1000.0

echo "Running Fusion with LaplaceLoss and vlbi_loss_weight 10000.0"
python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 10000.0
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 10000.0

echo "Running Fusion with LaplaceLoss and vlbi_sampling_weight 10.0"
python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 10.0
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 10.0

echo "Running Fusion with LaplaceLoss and vlbi_sampling_weight 100.0"
python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 100.0
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 100.0

echo "Running Fusion with LaplaceLoss and vlbi_sampling_weight 1000.0"
python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 1000.0
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 1000.0

echo "Running Fusion with LaplaceLoss and vlbi_sampling_weight 10000.0"
python src/train.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 10000.0
python src/inference_map.py --debug False --year 2023 --doy 010 --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 10000.0

# Wait for all background processes to finish
wait

echo "All training instances have completed."
