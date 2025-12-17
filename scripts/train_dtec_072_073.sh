#!/bin/bash
# Retry DTEC training for DOY 072 and 073 in parallel (failed due to bug, now fixed)

cd /scratch2/arrueegg/WP2/GIM_fusion_VLBI
source env/bin/activate

YEAR=2023
DAYS=(72 73)

echo "================================================================================"
echo "TRAINING DTEC FOR DOY 072 AND 073 (PARALLEL)"
echo "================================================================================"
echo ""

# Function to train a single day
train_dtec() {
    local doy=$1
    local file_doy=$(printf "%03d" $doy)
    
    echo "[$(date '+%H:%M:%S')] Training DTEC for DOY $doy"
    echo "Training DTEC..."
    python src/train.py --year $YEAR --doy $doy --mode DTEC_Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 100.0
    
    if [ $? -eq 0 ]; then
        echo "Running DTEC inference..."
        python src/inference_map.py --year $YEAR --doy $doy --mode DTEC_Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 100.0
        echo "[$(date '+%H:%M:%S')] ✓ Completed DOY $doy"
    else
        echo "[$(date '+%H:%M:%S')] ✗ Failed DOY $doy"
    fi
}

# Train both days in parallel
for doy in "${DAYS[@]}"; do
    train_dtec $doy &
done

# Wait for both jobs to complete
wait

echo ""
echo "================================================================================"
echo "DTEC TRAINING COMPLETED"
echo "================================================================================"
