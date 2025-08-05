#!/bin/bash

# Jason-3 Residual Analysis Runner
# This script provides an easy way to run the Jason-3 residual analysis for different approaches

set -e  # Exit on error

# Default values
APPROACH="GNSS"
YEAR=2023
OUTPUT_DIR="evaluation/compare_maps/jason3_residuals"
CONFIG="config/config.yaml"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Function to display help
show_help() {
    cat << EOF
Jason-3 Residual Analysis Runner

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -a, --approach APPROACH    Model approach to analyze (GNSS, Fusion, DTEC_Fusion)
                              Default: GNSS
    
    -y, --year YEAR           Year to process
                              Default: 2023
    
    -o, --output-dir DIR      Output directory for results
                              Default: evaluation/compare_maps/jason3_residuals
    
    -c, --config CONFIG       Configuration file path
                              Default: config/config.yaml
    
    --all-approaches         Run analysis for all approaches (GNSS, Fusion, DTEC_Fusion)
    
    --demo                   Run quick demo mode
    
    --compare                Compare results from different approaches
    
    -h, --help               Show this help message

EXAMPLES:
    # Analyze GNSS approach for 2023
    $0 --approach GNSS --year 2023
    
    # Analyze Fusion approach with custom output directory
    $0 --approach Fusion --output-dir my_results
    
    # Run analysis for all approaches
    $0 --all-approaches
    
    # Quick demo
    $0 --demo
    
    # Compare existing results
    $0 --compare

EOF
}

# Function to check if required files exist
check_requirements() {
    echo "Checking requirements..."
    
    # Check if we're in the right directory
    if [[ ! -f "config/config.yaml" ]]; then
        echo "Error: config/config.yaml not found. Please run from repository root."
        exit 1
    fi
    
    # Check Python script exists
    if [[ ! -f "evaluation/compare_maps/plot_jason3_residuals.py" ]]; then
        echo "Error: plot_jason3_residuals.py not found."
        exit 1
    fi
    
    echo "✓ Requirements check passed"
}

# Function to run analysis for a single approach
run_single_analysis() {
    local approach=$1
    local year=$2
    local output_dir=$3
    local config=$4
    
    echo "Running analysis for $approach approach (year: $year)..."
    echo "Output directory: $output_dir"
    
    cd "$REPO_ROOT"
    
    python evaluation/compare_maps/plot_jason3_residuals.py \
        --approach "$approach" \
        --year "$year" \
        --output-dir "$output_dir" \
        --config "$config" \
        --save-plots
    
    if [[ $? -eq 0 ]]; then
        echo "✓ Analysis completed successfully for $approach"
        echo "Results saved to: $output_dir"
    else
        echo "✗ Analysis failed for $approach"
        return 1
    fi
}

# Function to run analysis for all approaches
run_all_approaches() {
    local year=$1
    local base_output_dir=$2
    local config=$3
    
    local approaches=("GNSS" "Fusion" "DTEC_Fusion")
    local failed_approaches=()
    
    echo "Running analysis for all approaches..."
    
    for approach in "${approaches[@]}"; do
        local output_dir="${base_output_dir}_${approach,,}"
        
        echo ""
        echo "=" * 50
        echo "Processing: $approach"
        echo "=" * 50
        
        if run_single_analysis "$approach" "$year" "$output_dir" "$config"; then
            echo "✓ $approach completed successfully"
        else
            echo "✗ $approach failed"
            failed_approaches+=("$approach")
        fi
    done
    
    echo ""
    echo "Summary:"
    echo "--------"
    
    if [[ ${#failed_approaches[@]} -eq 0 ]]; then
        echo "✓ All approaches completed successfully"
    else
        echo "✗ Failed approaches: ${failed_approaches[*]}"
        echo "✓ Successful approaches: $((${#approaches[@]} - ${#failed_approaches[@]}))/${#approaches[@]}"
    fi
}

# Function to run demo mode
run_demo() {
    echo "Running demo mode..."
    cd "$REPO_ROOT"
    
    python evaluation/compare_maps/example_usage.py --mode demo
}

# Function to compare results
run_comparison() {
    echo "Comparing results from different approaches..."
    cd "$REPO_ROOT"
    
    python evaluation/compare_maps/example_usage.py --mode compare
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--approach)
            APPROACH="$2"
            shift 2
            ;;
        -y|--year)
            YEAR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        --all-approaches)
            ALL_APPROACHES=true
            shift
            ;;
        --demo)
            DEMO_MODE=true
            shift
            ;;
        --compare)
            COMPARE_MODE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Change to repository root
cd "$REPO_ROOT"

# Check requirements
check_requirements

# Run based on mode
if [[ "$DEMO_MODE" == true ]]; then
    run_demo
elif [[ "$COMPARE_MODE" == true ]]; then
    run_comparison
elif [[ "$ALL_APPROACHES" == true ]]; then
    run_all_approaches "$YEAR" "$OUTPUT_DIR" "$CONFIG"
else
    run_single_analysis "$APPROACH" "$YEAR" "$OUTPUT_DIR" "$CONFIG"
fi

echo ""
echo "Analysis complete!"
echo "Check the output directory for results and visualizations."
