#!/bin/bash
# Run SA evaluation for newly trained models

cd /scratch2/arrueegg/WP2/GIM_fusion_VLBI
source env/bin/activate

# Days that were trained (GNSS + VTEC)
DAYS_GNSS_VTEC=(19 34 47 74 89 103 110 142 173 187 194 198 296 338)

# Days that were trained (DTEC only)
DAYS_DTEC=(72 73)

YEAR=2023

echo "================================================================================"
echo "RUNNING SA EVALUATION FOR NEWLY TRAINED MODELS"
echo "================================================================================"
echo ""

# Run SA evaluation for GNSS + VTEC days
for doy in "${DAYS_GNSS_VTEC[@]}"; do
    file_doy=$(printf "%03d" $doy)
    echo "[$(date '+%H:%M:%S')] Running SA evaluation for DOY $doy"
    python src/eval_with_SA.py --year $YEAR --doy $doy
    echo "[$(date '+%H:%M:%S')] Completed DOY $doy"
done

# Run SA evaluation for DTEC days
for doy in "${DAYS_DTEC[@]}"; do
    file_doy=$(printf "%03d" $doy)
    echo "[$(date '+%H:%M:%S')] Running SA evaluation for DOY $doy"
    python src/eval_with_SA.py --year $YEAR --doy $doy
    echo "[$(date '+%H:%M:%S')] Completed DOY $doy"
done

echo ""
echo "================================================================================"
echo "SA EVALUATION COMPLETE"
echo "================================================================================"
