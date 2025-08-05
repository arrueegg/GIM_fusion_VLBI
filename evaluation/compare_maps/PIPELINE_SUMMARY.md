# Jason-3 Residual Analysis Pipeline - Summary

## Overview

I've created a comprehensive pipeline that allows you to plot residuals of any approach (GNSS, Fusion, DTEC_Fusion) against Jason-3 altimetry observations for 2023. The pipeline processes all Jason-3 observations, runs inference on the corresponding day's trained models, and stores results in a 1×1 degree global grid.

## What the Pipeline Does

1. **Loads Jason-3 Data**: Reads all Jason-3 altimetry observations for 2023 from the `sa_dataset.csv` file
2. **Model Inference**: For each observation:
   - Determines the day of year (DOY)
   - Loads the trained ensemble models for that specific day
   - Prepares model inputs (SM coordinates + temporal features)
   - Runs ensemble inference to get VTEC predictions
3. **Grid Storage**: Stores predictions, ground truth, and residuals in a 1×1 degree global grid
4. **Visualization**: Creates comprehensive plots including:
   - Global residuals map
   - Observation density map
   - Statistical analysis plots (histograms, scatter plots, etc.)

## Files Created

### Main Pipeline
- **`plot_jason3_residuals.py`**: The main pipeline script with the `Jason3ResidualPipeline` class
- **`run_analysis.sh`**: Shell script for easy execution with command-line options
- **`validate_pipeline.py`**: Validation script to check if everything is set up correctly
- **`example_usage.py`**: Example usage scripts and demonstrations

### Documentation  
- **`README.md`**: Comprehensive documentation with usage examples and troubleshooting

## Quick Start

### 1. Validate Setup
```bash
python evaluation/compare_maps/validate_pipeline.py
```

### 2. Run Analysis
```bash
# Option A: Using Python directly
python evaluation/compare_maps/plot_jason3_residuals.py --approach GNSS --year 2023

# Option B: Using shell script (recommended)
bash evaluation/compare_maps/run_analysis.sh --approach GNSS --year 2023

# Option C: Run all approaches
bash evaluation/compare_maps/run_analysis.sh --all-approaches
```

### 3. Demo Mode
```bash
# Quick demo to check functionality
bash evaluation/compare_maps/run_analysis.sh --demo
```

## Key Features

### ✅ **Automated Processing**
- Processes all Jason-3 observations for 2023 automatically
- Handles missing days/models gracefully
- Fault-tolerant design continues processing even if some days fail

### ✅ **Ensemble Support**
- Automatically detects and loads multiple model checkpoints per day
- Performs ensemble averaging for robust predictions
- Handles different model file formats

### ✅ **Flexible Model Support**
- Works with GNSS, Fusion, and DTEC_Fusion approaches
- Automatically detects experiment directory naming patterns
- Tries multiple common naming conventions

### ✅ **Grid-Based Output**
- 1×1 degree global grid resolution (180×360 cells)
- Handles multiple observations per grid cell with averaging
- Stores predictions, ground truth, residuals, and observation counts

### ✅ **Comprehensive Visualization**
- Global residuals map with coastlines and geographic features
- Observation density visualization
- Statistical plots (histograms, scatter plots, correlation analysis)
- Publication-ready figures with proper formatting

### ✅ **Data Formats**
- Saves results in multiple formats (`.npy`, `.npz`)
- Easy to load and analyze results programmatically
- Coordinate grids included for spatial analysis

## Expected Directory Structure

```
experiments/
├── GNSS_2023_001_SW1_LW1/model/           # Day 1 GNSS models
├── GNSS_2023_002_SW1_LW1/model/           # Day 2 GNSS models
├── Fusion_2023_001_SW1_LW1000/model/      # Day 1 Fusion models  
├── DTEC_Fusion_2023_001_SW1_LW100/model/  # Day 1 DTEC models
└── ...

data/
└── sa_dataset.csv                         # Jason-3 observations

evaluation/compare_maps/
├── plot_jason3_residuals.py              # Main pipeline
├── run_analysis.sh                       # Shell runner
├── validate_pipeline.py                  # Validation script
├── example_usage.py                      # Usage examples
└── README.md                             # Documentation
```

## Output Files

For each approach, the pipeline generates:

### Data Files
- `{approach}_{year}_predictions.npy` - Model predictions grid
- `{approach}_{year}_ground_truth.npy` - Ground truth VTEC grid
- `{approach}_{year}_residuals.npy` - Residuals grid
- `{approach}_{year}_counts.npy` - Observation counts per grid cell
- `{approach}_{year}_results.npz` - Complete compressed results

### Visualization Files
- `{approach}_{year}_residuals_map.png` - Global residuals map
- `jason3_{year}_observation_density.png` - Observation density
- `{approach}_{year}_statistics.png` - Statistical analysis plots

## Pipeline Architecture

The `Jason3ResidualPipeline` class handles:

1. **Data Loading**: Loads and filters Jason-3 observations
2. **Model Management**: Finds and loads trained models for each day
3. **Coordinate Processing**: Handles GEO↔SM coordinate transformations
4. **Temporal Features**: Extracts time-based features (SOD, sin/cos UTC)
5. **Ensemble Inference**: Runs inference with multiple models and averages
6. **Grid Management**: Updates 1×1 degree global grid with running averages
7. **Visualization**: Creates publication-ready plots and maps

## Error Handling

The pipeline is designed to be robust:

- **Missing Models**: Logs warning and continues with available days
- **Data Issues**: Handles missing/corrupted data gracefully  
- **Memory Management**: Processes data day-by-day to avoid memory issues
- **Device Selection**: Automatically uses GPU if available, falls back to CPU
- **Path Resolution**: Tries multiple common paths for data and models

## Performance

- **Processing Time**: ~10-30 minutes for full year (depends on hardware/models available)
- **Memory Usage**: Moderate (processes day-by-day)
- **GPU Acceleration**: Automatic GPU usage for faster inference
- **Parallel Processing**: Can be extended for parallel day processing

## Next Steps

1. **Run Validation**: Check that everything is set up correctly
2. **Start with Demo**: Try the demo mode to verify functionality
3. **Single Approach**: Run analysis for one approach (e.g., GNSS)
4. **Compare Approaches**: Run for multiple approaches and compare results
5. **Analyze Results**: Load saved grids and perform detailed analysis

## Support

The pipeline includes comprehensive error messages and logging. If you encounter issues:

1. Run `validate_pipeline.py` to identify setup problems
2. Check the README.md for troubleshooting guidance
3. Use demo mode to test with minimal data
4. Review log messages for specific error details

The pipeline is designed to be self-contained and should work with your existing repository structure and data formats.
