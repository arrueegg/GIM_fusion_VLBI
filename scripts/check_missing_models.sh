#!/bin/bash
# Check which models are missing for days that should have models (based on Unified_doy_list.txt)

cd /scratch2/arrueegg/WP2/GIM_fusion_VLBI

YEAR=2023
EXPERIMENTS_DIR="experiments"
DOY_LIST_FILE="scripts/lists/Unified_doy_list.txt"

if [ ! -f "$DOY_LIST_FILE" ]; then
    echo "Error: DOY list file not found: $DOY_LIST_FILE"
    exit 1
fi

# Extract DOY list for 2023
EXPECTED_DAYS=($(grep "^${YEAR} " "$DOY_LIST_FILE" | awk '{print $2}'))

echo "================================================================================"
echo "CHECKING MODEL AVAILABILITY FOR YEAR $YEAR"
echo "================================================================================"
echo "Expected days with models: ${#EXPECTED_DAYS[@]}"
echo ""

# All 3 methods we need
METHODS=("GNSS_${YEAR}_XXX_SW1_LW1" "DTEC_Fusion_${YEAR}_XXX_SW1_LW100" "Fusion_${YEAR}_XXX_SW1_LW1000")

missing_days=()

# Check only expected DOYs from Unified_doy_list.txt
for doy in "${EXPECTED_DAYS[@]}"; do
    file_doy=$(printf "%03d" $doy)
    missing_methods=()
    
    for method_template in "${METHODS[@]}"; do
        method="${method_template/XXX/$file_doy}"
        model_dir="${EXPERIMENTS_DIR}/${method}/model"
        
        # Check if model directory exists and is not empty
        if [ ! -d "$model_dir" ] || [ -z "$(ls -A $model_dir 2>/dev/null)" ]; then
            # Extract method name for display
            method_name=$(echo $method | cut -d'_' -f1-2)
            missing_methods+=("$method_name")
        fi
    done
    
    # If any methods are missing, report it
    if [ ${#missing_methods[@]} -gt 0 ]; then
        missing_days+=("$file_doy")
        echo "DOY $file_doy: Missing ${missing_methods[*]}"
    fi
done

echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo "Total expected days: ${#EXPECTED_DAYS[@]}"
echo "Days with all 3 models: $((${#EXPECTED_DAYS[@]} - ${#missing_days[@]}))"
echo "Days with missing models: ${#missing_days[@]}"

if [ ${#missing_days[@]} -gt 0 ]; then
    echo "Days: ${missing_days[*]}"
else
    echo "All models are complete for all ${#EXPECTED_DAYS[@]} expected days!"
fi
echo "================================================================================"
