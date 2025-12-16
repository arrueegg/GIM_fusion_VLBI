#!/bin/bash
# Train missing models in parallel on local machine

cd /scratch2/arrueegg/WP2/GIM_fusion_VLBI
source env/bin/activate

# Days that need GNSS and VTEC training (have VLBI data, only DTEC exists)
DAYS_NEED_GNSS_VTEC=(19 34 47 74 89 103 110 142 173 187 194 198 296 338)

# Days that need DTEC training (have VLBI data, only GNSS/VTEC exist)
DAYS_NEED_DTEC=(72 73)

YEAR=2023
MAX_PARALLEL=4  # Number of days to train in parallel (adjust based on GPU memory)

echo "================================================================================"
echo "TRAINING MISSING MODELS IN PARALLEL"
echo "================================================================================"
echo ""
echo "Days needing GNSS + VTEC: ${#DAYS_NEED_GNSS_VTEC[@]}"
echo "Days needing DTEC: ${#DAYS_NEED_DTEC[@]}"
echo "Max parallel jobs: $MAX_PARALLEL"
echo ""
echo "Note: Days 072, 073, 108, 186, 233, 326 have NO VLBI data and are excluded"
echo ""

mkdir -p logs/parallel_training

# Function to train GNSS and VTEC for a single day
train_gnss_vtec() {
    local doy=$1
    local file_doy=$(printf "%03d" $doy)
    local log_file="logs/parallel_training/train_${YEAR}_${file_doy}_GNSS_VTEC.log"
    
    echo "[$(date '+%H:%M:%S')] Starting GNSS + VTEC training for DOY $doy"
    
    {
        echo "================================================================================"
        echo "Training GNSS and VTEC for ${YEAR}_${file_doy}"
        echo "================================================================================"
        
        # Train GNSS
        if [ ! -d "experiments/GNSS_${YEAR}_${file_doy}_SW1_LW1/model" ] || [ -z "$(ls -A experiments/GNSS_${YEAR}_${file_doy}_SW1_LW1/model 2>/dev/null)" ]; then
            echo "Training GNSS..."
            python src/train.py --year $YEAR --doy $doy --mode GNSS --loss_fn LaplaceLoss
            echo "Running GNSS inference..."
            python src/inference_map.py --year $YEAR --doy $doy --mode GNSS --loss_fn LaplaceLoss
        else
            echo "GNSS model exists, skipping"
        fi
        
        # Train VTEC (Fusion with LW=1000)
        if [ ! -d "experiments/Fusion_${YEAR}_${file_doy}_SW1_LW1000/model" ] || [ -z "$(ls -A experiments/Fusion_${YEAR}_${file_doy}_SW1_LW1000/model 2>/dev/null)" ]; then
            echo "Training VTEC (Fusion)..."
            python src/train.py --year $YEAR --doy $doy --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 1000.0
            echo "Running VTEC inference..."
            python src/inference_map.py --year $YEAR --doy $doy --mode Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 1000.0
        else
            echo "VTEC model exists, skipping"
        fi
        
        echo "Completed ${YEAR}_${file_doy}"
        
    } > "$log_file" 2>&1
    
    echo "[$(date '+%H:%M:%S')] Completed DOY $doy (log: $log_file)"
}

# Function to train DTEC for a single day
train_dtec() {
    local doy=$1
    local file_doy=$(printf "%03d" $doy)
    local log_file="logs/parallel_training/train_${YEAR}_${file_doy}_DTEC.log"
    
    echo "[$(date '+%H:%M:%S')] Starting DTEC training for DOY $doy"
    
    {
        echo "================================================================================"
        echo "Training DTEC for ${YEAR}_${file_doy}"
        echo "================================================================================"
        
        if [ ! -d "experiments/DTEC_Fusion_${YEAR}_${file_doy}_SW1_LW100/model" ] || [ -z "$(ls -A experiments/DTEC_Fusion_${YEAR}_${file_doy}_SW1_LW100/model 2>/dev/null)" ]; then
            echo "Training DTEC..."
            python src/train.py --year $YEAR --doy $doy --mode DTEC_Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 100.0
            echo "Running DTEC inference..."
            python src/inference_map.py --year $YEAR --doy $doy --mode DTEC_Fusion --loss_fn LaplaceLoss --vlbi_loss_weight 100.0
        else
            echo "DTEC model exists, skipping"
        fi
        
        echo "Completed ${YEAR}_${file_doy}"
        
    } > "$log_file" 2>&1
    
    echo "[$(date '+%H:%M:%S')] Completed DOY $doy (log: $log_file)"
}

# Train GNSS + VTEC for days that need them
echo "================================================================================"
echo "PHASE 1: Training GNSS + VTEC (${#DAYS_NEED_GNSS_VTEC[@]} days)"
echo "================================================================================"

job_count=0
for doy in "${DAYS_NEED_GNSS_VTEC[@]}"; do
    train_gnss_vtec $doy &
    ((job_count++))
    
    # Wait if we've reached max parallel jobs
    if [ $job_count -ge $MAX_PARALLEL ]; then
        wait -n  # Wait for any one job to finish
        ((job_count--))
    fi
done

# Wait for remaining GNSS+VTEC jobs
wait

echo ""
echo "================================================================================"
echo "PHASE 2: Training DTEC (${#DAYS_NEED_DTEC[@]} days)"
echo "================================================================================"

# Train DTEC for days that need it
for doy in "${DAYS_NEED_DTEC[@]}"; do
    train_dtec $doy &
done

# Wait for DTEC jobs
wait

echo ""
echo "================================================================================"
echo "ALL TRAINING COMPLETED"
echo "================================================================================"
echo "Check logs in: logs/parallel_training/"
echo ""
echo "Next step: Run SA evaluation for all days"
echo "  python src/eval_with_SA.py --year 2023 --doy XXX"
echo "================================================================================"
