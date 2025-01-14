#!/bin/bash

cd /cluster/work/igp_psr/arrueegg/WP2/GIM_fusion_VLBI/scripts/
source ../env/bin/activate

# Path to the DOY list file
ls
DOY_LIST_FILE="lists/Unified_doy_list.txt"

# Check if the DOY list file exists
if [[ ! -f "$DOY_LIST_FILE" ]]; then
    echo "DOY list file not found: $DOY_LIST_FILE"
    exit 1
fi

# Check if the correct number of arguments are provided
if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <start_year> <start_doy> <end_year> <end_doy>"
    exit 1
fi

START_YEAR=$1
START_DOY=$2
END_YEAR=$3
END_DOY=$4

# Iterate over each DOY in the list and call submit_cluster.sh if it matches the criteria
while IFS= read -r line; do
    line_year=$(echo "$line" | cut -d' ' -f1)
    line_doy=$(echo "$line" | cut -d' ' -f2)

    echo $line_year $line_doy 
    
    if [[ "$START_YEAR" -eq "$END_YEAR" ]]; then
        if [[ "$line_year" -eq "$START_YEAR" && "$line_doy" -ge "$START_DOY" && "$line_doy" -le "$END_DOY" ]]; then
            echo "submit: $line_year $line_doy"
            sbatch submit_cluster.sh "$line_year $line_doy"
        fi
    else
        if [[ "$line_year" -gt "$START_YEAR" && "$line_year" -lt "$END_YEAR" ]]; then
            echo "submit: $line_year $line_doy"
            sbatch submit_cluster.sh "$line_year $line_doy"
        elif [[ "$line_year" -eq "$START_YEAR" && "$line_doy" -ge "$START_DOY" ]]; then
            echo "submit: $line_year $line_doy"
            sbatch submit_cluster.sh "$line_year $line_doy"
        elif [[ "$line_year" -eq "$END_YEAR" && "$line_doy" -le "$END_DOY" ]]; then
            echo "submit: $line_year $line_doy"
            sbatch submit_cluster.sh "$line_year $line_doy"
        fi
    fi
done < "$DOY_LIST_FILE"