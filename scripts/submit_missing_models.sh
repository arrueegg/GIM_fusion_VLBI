#!/bin/bash
# Submit cluster jobs to train missing GNSS and VTEC models
# Only trains days that have VLBI data but are missing GNSS/VTEC models

# Days that need GNSS and VTEC training (have VLBI data, only DTEC exists)
DAYS_NEED_GNSS_VTEC=(19 34 47 74 89 103 110 142 173 187 194 198 296 338)

# Days that need DTEC training (have VLBI data, only GNSS/VTEC exist)
DAYS_NEED_DTEC=(72 73)

YEAR=2023

echo "================================================================================"
echo "SUBMITTING CLUSTER JOBS FOR MISSING MODELS"
echo "================================================================================"
echo ""
echo "Days needing GNSS + VTEC models: ${#DAYS_NEED_GNSS_VTEC[@]}"
echo "Days needing DTEC models: ${#DAYS_NEED_DTEC[@]}"
echo ""
echo "Note: Days 072, 073, 108, 186, 233, 326 have NO VLBI data and are excluded"
echo ""

cd /scratch2/arrueegg/WP2/GIM_fusion_VLBI/scripts/

# Submit jobs for days needing GNSS and VTEC
for doy in "${DAYS_NEED_GNSS_VTEC[@]}"; do
    echo "Submitting GNSS + VTEC training for DOY $doy"
    sbatch submit_cluster_gnss_vtec.sh $YEAR $doy
done

# Submit jobs for days needing DTEC
for doy in "${DAYS_NEED_DTEC[@]}"; do
    echo "Submitting DTEC training for DOY $doy"
    sbatch submit_cluster_dtec.sh $YEAR $doy
done

echo ""
echo "All jobs submitted. Check status with: squeue -u $USER"
echo "================================================================================"
