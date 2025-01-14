#!/bin/bash

# assign the first argument to the $YEAR variable
YEAR=$1
# assign the second argument to the $DOY variable
DOY=$2
# assign the third argument to the $CONFIG_PATH variable if provided, otherwise use default config/config.yaml
CONFIG_PATH=${3:-config/config.yaml}

#cd /scratch2/arrueegg/WP2/GIM_fusion_VLBI/
#source env/bin/activate

# echo "Running GNSS with MSE"
# python src/train.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode GNSS --loss_fn MSELoss
# python src/inference_map.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode GNSS --loss_fn MSELoss
# python src/eval_with_SA.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode GNSS --loss_fn MSELoss

echo "Running GNSS with LaplaceLoss"
python src/train.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode GNSS --loss_fn LaplaceLoss
python src/inference_map.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode GNSS --loss_fn LaplaceLoss
python src/eval_with_SA.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode GNSS --loss_fn LaplaceLoss

# echo "Running Fusion with LaplaceLoss"
# python src/train.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode Fusion --loss_fn LaplaceLoss
# python src/inference_map.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode Fusion --loss_fn LaplaceLoss
# python src/eval_with_SA.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode Fusion --loss_fn LaplaceLoss

echo "Running Fusion with LaplaceLoss and vlbi_sampling_weight 1000.0"
python src/train.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 1000.0
python src/inference_map.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 1000.0
python src/eval_with_SA.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 1000.0

echo "Running Fusion with LaplaceLoss and vlbi_loss_weight 1000.0"
python src/train.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 1000.0
python src/inference_map.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 1000.0
python src/eval_with_SA.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 1000.0

# echo "Running DTEC Fusion with LaplaceLoss"
# python src/train.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode DTEC_Fusion --loss_fn LaplaceLoss
# python src/inference_map.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode DTEC_Fusion --loss_fn LaplaceLoss
# python src/eval_with_SA.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode DTEC_Fusion --loss_fn LaplaceLoss

echo "Running DTEC Fusion with LaplaceLoss and vlbi_loss_weight 100.0"
python src/train.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode DTEC_Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 100.0
python src/inference_map.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode DTEC_Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 100.0
python src/eval_with_SA.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode DTEC_Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 100.0

echo "Running DTEC Fusion with LaplaceLoss and vlbi_sampling_weight 100.0"
python src/train.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode DTEC_Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 100.0
python src/inference_map.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode DTEC_Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 100.0
python src/eval_with_SA.py --debug False --config_path $CONFIG_PATH --year $YEAR --doy $DOY --mode DTEC_Fusion --loss_fn LaplaceLoss --vlbi_sampling_weight 100.0

# Wait for all background processes to finish
wait

echo "All training instances have completed."
