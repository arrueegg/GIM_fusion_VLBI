#!/usr/bin/env python3
"""
Jason-3 Residuals Plotting Pipeline

This script creates a pipeline to plot bias-corrected residuals of an approach (e.g., GNSS) 
to Jason-3 observations. It processes all Jason-3 altimetry observations in 2023, runs 
inference on the corresponding day's model, applies daily bias correction (removes global 
mean deviation), and stores predictions, ground truth, and bias-corrected residuals in a 1x1 degree grid.

Bias Correction:
- For each day, calculates the global mean deviation: bias = mean(predictions - observations)
- Subtracts this daily bias from predictions before computing residuals
- This removes systematic daily offsets while preserving spatial patterns

Author: GitHub Copilot
Date: 2025-01-03
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.locationencoder.pe import SphericalHarmonics
from utils.config_parser import parse_config
from models.model import get_model
from spacepy.coordinates import Coords
from spacepy.time import Ticktock

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Jason3ResidualPipeline:
    """
    Pipeline for processing Jason-3 observations and computing bias-corrected model residuals.
    
    The pipeline applies daily bias correction by computing the global mean deviation 
    between model predictions and Jason-3 observations for each day, then subtracting 
    this bias from residuals before storing them in the 1x1 degree grid.
    """
    
    def __init__(self, config, approach="GNSS", year=2023):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
            approach: Model approach to use (e.g., "GNSS", "Fusion", "DTEC_Fusion")
            year: Year to process (default: 2023)
        """
        self.config = config
        self.approach = approach
        self.year = year
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up spherical harmonics encoder if needed
        self.sh_encoder = None
        if config['preprocessing'].get('SH_encoding'):
            self.sh_encoder = SphericalHarmonics(config['preprocessing']['SH_degree']).to(self.device)
        
        # Initialize results grid (1x1 degree resolution: 180x360)
        self.lat_bins = np.arange(-89.5, 90.5, 1.0)  # Center of 1-degree bins
        self.lon_bins = np.arange(-179.5, 180.5, 1.0)
        self.results_grid = {
            'predictions': np.full((len(self.lat_bins), len(self.lon_bins)), np.nan),
            'ground_truth': np.full((len(self.lat_bins), len(self.lon_bins)), np.nan),
            'residuals': np.full((len(self.lat_bins), len(self.lon_bins)), np.nan),
            'counts': np.zeros((len(self.lat_bins), len(self.lon_bins))),
            'lat_grid': self.lat_bins,
            'lon_grid': self.lon_bins
        }
        
    def load_jason3_data(self):
        """
        Load all Jason-3 observations for the specified year.
        
        Returns:
            pd.DataFrame: Jason-3 observations with columns [lon, lat, time, vtec, sm_lat, sm_lon]
        """
        logger.info(f"Loading Jason-3 data for year {self.year}")
        
        # Path to the SA dataset (as used in eval_with_SA.py)
        csv_path = os.path.join(self.config['data']['GNSS_data_path'], "sa_dataset.csv")
        
        # Alternative paths for different environments
        if 'cluster' in csv_path or not os.path.exists(csv_path):
            alternative_paths = [
                '/cluster/work/igp_psr/arrueegg/sa_dataset.csv',
                '/home/space/internal/ggltmp/4Arno/sa_dataset.csv',
                '/scratch2/arrueegg/WP2/GIM_fusion_VLBI/data/sa_dataset.csv'
            ]
            
            csv_path = None
            for path in alternative_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
        
        if csv_path is None or not os.path.exists(csv_path):
            raise FileNotFoundError(
                "Could not find sa_dataset.csv. Please ensure Jason-3 data is available. "
                "Expected locations: " + str(alternative_paths)
            )
        
        logger.info(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        df['time'] = pd.to_datetime(df['time'])
        
        # Filter for the specified year
        df = df[df['time'].dt.year == self.year].copy()
        
        if df.empty:
            raise ValueError(f"No Jason-3 data found for year {self.year}")
        
        logger.info(f"Loaded {len(df)} Jason-3 observations for year {self.year}")
        
        # Ensure longitude is in [-180, 180] range
        df['lon'] = ((df['lon'] + 180) % 360) - 180
        
        return df
    
    def get_model_path(self, doy):
        """
        Get the path to the trained model for a specific day of year.
        
        Args:
            doy: Day of year
            
        Returns:
            str: Path to model directory, or None if not found
        """
        # Construct experiment directory based on approach
        if self.approach == "GNSS":
            exp_dir = f"GNSS_{self.year}_{doy:03d}_SW1_LW1"
        elif self.approach == "Fusion":
            exp_dir = f"Fusion_{self.year}_{doy:03d}_SW1_LW1000"  # Most common fusion variant
        elif self.approach == "DTEC_Fusion":
            exp_dir = f"DTEC_Fusion_{self.year}_{doy:03d}_SW1_LW100"  # Most common DTEC variant
        else:
            # Generic approach
            exp_dir = f"{self.approach}_{self.year}_{doy:03d}_SW1_LW1"
        
        exp_path = os.path.join("experiments", exp_dir)
        model_dir = os.path.join(exp_path, "model")
        
        if os.path.exists(model_dir) and os.listdir(model_dir):
            return model_dir
        
        # Try alternative paths
        alt_exp_dirs = [
            f"{self.approach}_{self.year}_{doy:03d}_SW1_LW100",
            f"{self.approach}_{self.year}_{doy:03d}_SW100_LW1",
            f"{self.approach}_{self.year}_{doy:03d}_SW1000_LW1",
            f"{self.approach}_{self.year}_{doy:03d}",
        ]
        
        for alt_dir in alt_exp_dirs:
            alt_path = os.path.join("experiments", alt_dir, "model")
            if os.path.exists(alt_path) and os.listdir(alt_path):
                return alt_path
        
        return None
    
    def load_model_ensemble(self, model_dir):
        """
        Load ensemble of trained models from a directory.
        
        Args:
            model_dir: Directory containing model checkpoint files
            
        Returns:
            list: List of loaded models
        """
        models = []
        
        for filename in os.listdir(model_dir):
            if filename.endswith('.pth'):
                model_path = os.path.join(model_dir, filename)
                
                try:
                    model = get_model(self.config).to(self.device)
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    models.append(model)
                except Exception as e:
                    logger.warning(f"Failed to load model {model_path}: {e}")
        
        return models
    
    def prepare_inputs(self, observations):
        """
        Prepare model inputs from Jason-3 observations.
        
        Args:
            observations: DataFrame with Jason-3 data
            
        Returns:
            torch.Tensor: Model inputs
        """
        # Use SM coordinates
        lat = torch.tensor(observations['sm_lat'].values, dtype=torch.float32, device=self.device)
        lon = torch.tensor(observations['sm_lon'].values, dtype=torch.float32, device=self.device)
        
        # Start with coordinate features
        inputs = torch.stack([lon, lat], dim=1)
        
        # Apply spherical harmonics encoding if configured
        if self.sh_encoder:
            inputs = self.sh_encoder(inputs)
        
        # Add temporal features
        times = (observations['time'].dt.hour * 3600 +
                observations['time'].dt.minute * 60 +
                observations['time'].dt.second)
        sod = torch.tensor(times.values, dtype=torch.float32, device=self.device)
        sin_utc = torch.sin(sod / 86400 * 2 * torch.pi)
        cos_utc = torch.cos(sod / 86400 * 2 * torch.pi)
        norm = 2 * sod / 86400 - 1
        
        # Concatenate all features
        inputs = torch.cat([
            inputs,
            sin_utc.unsqueeze(1),
            cos_utc.unsqueeze(1),
            norm.unsqueeze(1)
        ], dim=1)
        
        return inputs
    
    def run_inference(self, models, inputs):
        """
        Run ensemble inference on the inputs.
        
        Args:
            models: List of trained models
            inputs: Model inputs
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        predictions = []
        
        with torch.no_grad():
            for model in models:
                pred = model(inputs).cpu().numpy()
                # Extract VTEC predictions (first column if multiple outputs)
                if pred.ndim > 1 and pred.shape[1] > 1:
                    pred = pred[:, 0]
                predictions.append(pred.flatten())
        
        # Ensemble averaging
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def update_grid(self, observations, predictions, daily_bias=0.0):
        """
        Update the results grid with new observations and predictions.
        
        Args:
            observations: DataFrame with Jason-3 data
            predictions: Model predictions
            daily_bias: Daily global bias to correct for (predictions - ground_truth mean)
        """
        # Convert to grid indices
        lat_indices = np.digitize(observations['lat'].values, 
                                 np.arange(-90, 91, 1)) - 1
        lon_indices = np.digitize(observations['lon'].values,
                                 np.arange(-180, 181, 1)) - 1
        
        # Clip to valid range
        lat_indices = np.clip(lat_indices, 0, len(self.lat_bins) - 1)
        lon_indices = np.clip(lon_indices, 0, len(self.lon_bins) - 1)
        
        ground_truth = observations['vtec'].values
        # Apply bias correction to residuals
        bias_corrected_predictions = predictions - daily_bias
        residuals = bias_corrected_predictions - ground_truth
        
        # Update grid (accumulate values for averaging)
        for i, (lat_idx, lon_idx) in enumerate(zip(lat_indices, lon_indices)):
            current_count = self.results_grid['counts'][lat_idx, lon_idx]
            
            if current_count == 0:
                # First observation in this grid cell
                self.results_grid['predictions'][lat_idx, lon_idx] = bias_corrected_predictions[i]
                self.results_grid['ground_truth'][lat_idx, lon_idx] = ground_truth[i]
                self.results_grid['residuals'][lat_idx, lon_idx] = residuals[i]
            else:
                # Running average
                weight = 1.0 / (current_count + 1)
                self.results_grid['predictions'][lat_idx, lon_idx] = (
                    self.results_grid['predictions'][lat_idx, lon_idx] * (1 - weight) + 
                    bias_corrected_predictions[i] * weight
                )
                self.results_grid['ground_truth'][lat_idx, lon_idx] = (
                    self.results_grid['ground_truth'][lat_idx, lon_idx] * (1 - weight) + 
                    ground_truth[i] * weight
                )
                self.results_grid['residuals'][lat_idx, lon_idx] = (
                    self.results_grid['residuals'][lat_idx, lon_idx] * (1 - weight) + 
                    residuals[i] * weight
                )
            
            self.results_grid['counts'][lat_idx, lon_idx] += 1
    
    def process_all_observations(self):
        """
        Process all Jason-3 observations for the year.
        
        Returns:
            dict: Processing statistics
        """
        logger.info("Starting processing of all Jason-3 observations")
        
        # Load all Jason-3 data
        jason3_data = self.load_jason3_data()
        
        # Group by day of year
        jason3_data['doy'] = jason3_data['time'].dt.dayofyear
        grouped = jason3_data.groupby('doy')
        
        stats = {
            'total_days': len(grouped),
            'processed_days': 0,
            'total_observations': len(jason3_data),
            'processed_observations': 0,
            'failed_days': [],
            'model_not_found_days': [],
            'daily_biases': []  # Track daily bias values
        }
        
        logger.info(f"Found observations for {stats['total_days']} days")
        
        # Process each day
        for doy, day_data in tqdm(grouped, desc="Processing days"):
            try:
                # Check if model exists for this day
                model_dir = self.get_model_path(doy)
                if model_dir is None:
                    logger.warning(f"No model found for DOY {doy}")
                    stats['model_not_found_days'].append(doy)
                    continue
                
                # Load models for this day
                models = self.load_model_ensemble(model_dir)
                if not models:
                    logger.warning(f"No valid models found in {model_dir} for DOY {doy}")
                    stats['model_not_found_days'].append(doy)
                    continue
                
                # Prepare inputs
                inputs = self.prepare_inputs(day_data)
                
                # Run inference
                predictions = self.run_inference(models, inputs)
                
                # Calculate daily global bias (mean deviation)
                ground_truth = day_data['vtec'].values
                daily_bias = np.mean(predictions - ground_truth)
                
                logger.debug(f"DOY {doy}: Daily bias = {daily_bias:.3f} TECU")
                stats['daily_biases'].append(daily_bias)
                
                # Update grid with bias correction
                self.update_grid(day_data, predictions, daily_bias)
                
                stats['processed_days'] += 1
                stats['processed_observations'] += len(day_data)
                
                logger.debug(f"Processed DOY {doy}: {len(day_data)} observations")
                
            except Exception as e:
                logger.error(f"Failed to process DOY {doy}: {e}")
                stats['failed_days'].append(doy)
        
        logger.info(f"Processing complete. Processed {stats['processed_observations']} observations "
                   f"from {stats['processed_days']} days")
        
        return stats
    
    def save_results(self, output_dir):
        """
        Save the results grid to files.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as numpy arrays
        for key in ['predictions', 'ground_truth', 'residuals', 'counts']:
            filename = f"{self.approach}_{self.year}_{key}.npy"
            filepath = os.path.join(output_dir, filename)
            np.save(filepath, self.results_grid[key])
            logger.info(f"Saved {key} to {filepath}")
        
        # Save coordinate grids
        np.save(os.path.join(output_dir, f"{self.approach}_{self.year}_lat_grid.npy"), 
                self.lat_bins)
        np.save(os.path.join(output_dir, f"{self.approach}_{self.year}_lon_grid.npy"), 
                self.lon_bins)
        
        # Save as compressed npz file
        npz_path = os.path.join(output_dir, f"{self.approach}_{self.year}_results.npz")
        np.savez_compressed(npz_path, **self.results_grid)
        logger.info(f"Saved complete results to {npz_path}")
    
    def plot_results(self, output_dir, save_plots=True):
        """
        Create plots of the results.
        
        Args:
            output_dir: Directory to save plots
            save_plots: Whether to save plots to disk
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create lon/lat meshgrids for plotting
        lon_mesh, lat_mesh = np.meshgrid(self.lon_bins, self.lat_bins)
        
        # Plot 1: Residuals map
        fig = plt.figure(figsize=(15, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
        
        # Plot residuals
        residuals_masked = np.ma.masked_where(
            np.isnan(self.results_grid['residuals']) | (self.results_grid['counts'] == 0),
            self.results_grid['residuals']
        )
        
        border_low = np.nanpercentile(self.results_grid['residuals'], 5)
        border_high = np.nanpercentile(self.results_grid['residuals'], 95)
        border = max(abs(border_low), abs(border_high))
        im = ax.pcolormesh(lon_mesh, lat_mesh, residuals_masked, 
                          transform=ccrs.PlateCarree(),
                          cmap='RdBu_r', shading='nearest',
                          vmin=-border,
                          vmax=border)
        
        plt.colorbar(im, ax=ax, orientation='horizontal', 
                    shrink=0.6, pad=0.05, label='VTEC Residuals (TECU)')
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        plt.title(f'{self.approach} Model Residuals vs Jason-3 Observations ({self.year})\n'
                 f'1°×1° Grid Resolution', fontsize=14, pad=20)
        
        if save_plots:
            plt.savefig(os.path.join(output_dir, f'{self.approach}_{self.year}_residuals_map.png'),
                       dpi=300, bbox_inches='tight')
            logger.info(f"Saved residuals map to {output_dir}")
                
        # Plot 2: Observation count map
        fig = plt.figure(figsize=(15, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
        
        counts_masked = np.ma.masked_where(self.results_grid['counts'] == 0,
                                          self.results_grid['counts'])
        
        im = ax.pcolormesh(lon_mesh, lat_mesh, counts_masked,
                          transform=ccrs.PlateCarree(),
                          cmap='viridis', shading='nearest')
        
        plt.colorbar(im, ax=ax, orientation='horizontal',
                    shrink=0.6, pad=0.05, label='Number of Observations')
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        plt.title(f'Jason-3 Observation Density ({self.year})\n'
                 f'1°×1° Grid Resolution', fontsize=14, pad=20)
        
        if save_plots:
            plt.savefig(os.path.join(output_dir, f'jason3_{self.year}_observation_density.png'),
                       dpi=300, bbox_inches='tight')
            logger.info(f"Saved observation density map to {output_dir}")
                
        # Plot 3: Statistics histogram
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Residuals histogram
        valid_residuals = self.results_grid['residuals'][~np.isnan(self.results_grid['residuals'])]
        axes[0].hist(valid_residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Residuals (TECU)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Residuals Distribution')
        axes[0].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        # Add statistics text
        rmse = np.sqrt(np.mean(valid_residuals**2))
        mae = np.mean(np.abs(valid_residuals))
        bias = np.mean(valid_residuals)
        axes[0].text(0.05, 0.95, f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nBias: {bias:.2f}',
                    transform=axes[0].transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Predictions vs ground truth
        valid_mask = ~np.isnan(self.results_grid['predictions']) & ~np.isnan(self.results_grid['ground_truth'])
        pred_flat = self.results_grid['predictions'][valid_mask]
        truth_flat = self.results_grid['ground_truth'][valid_mask]
        
        axes[1].scatter(truth_flat, pred_flat, alpha=0.5, s=1)
        min_val, max_val = min(truth_flat.min(), pred_flat.min()), max(truth_flat.max(), pred_flat.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        axes[1].set_xlabel('Ground Truth (TECU)')
        axes[1].set_ylabel('Predictions (TECU)')
        axes[1].set_title('Predictions vs Ground Truth')
        
        # Correlation
        correlation = np.corrcoef(pred_flat, truth_flat)[0, 1]
        axes[1].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                    transform=axes[1].transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Observation counts histogram
        valid_counts = self.results_grid['counts'][self.results_grid['counts'] > 0]
        axes[2].hist(valid_counts, bins=50, alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('Observations per Grid Cell')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Observation Count Distribution')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(output_dir, f'{self.approach}_{self.year}_statistics.png'),
                       dpi=300, bbox_inches='tight')
            logger.info(f"Saved statistics plots to {output_dir}")
        

def main():
    """Main function to run the Jason-3 residual analysis pipeline."""
    parser = argparse.ArgumentParser(description='Jason-3 Residual Analysis Pipeline')
    parser.add_argument('--approach', type=str, default='GNSS',
                       choices=['GNSS', 'Fusion', 'DTEC_Fusion'],
                       help='Model approach to analyze')
    parser.add_argument('--year', type=int, default=2023,
                       help='Year to process')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--output-dir', type=str, 
                       default='evaluation/compare_maps/jason3_residuals',
                       help='Output directory for results')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save plots to files')
    
    args = parser.parse_args()
    
    # Change to repository root directory
    repo_root = Path(__file__).parent.parent.parent
    os.chdir(repo_root)
    
    logger.info(f"Starting Jason-3 residual analysis for {args.approach} approach, year {args.year}")
    
    try:
        # Load configuration
        config = parse_config()
        
        # Initialize pipeline
        pipeline = Jason3ResidualPipeline(config, args.approach, args.year)
        
        # Process all observations
        stats = pipeline.process_all_observations()
        
        # Print processing statistics
        logger.info("Processing Statistics:")
        logger.info(f"  Total days with data: {stats['total_days']}")
        logger.info(f"  Successfully processed days: {stats['processed_days']}")
        logger.info(f"  Total observations: {stats['total_observations']}")
        logger.info(f"  Processed observations: {stats['processed_observations']}")
        logger.info(f"  Days with missing models: {len(stats['model_not_found_days'])}")
        logger.info(f"  Failed days: {len(stats['failed_days'])}")
        
        # Print bias correction statistics
        if stats['daily_biases']:
            daily_biases = np.array(stats['daily_biases'])
            logger.info(f"Bias Correction Statistics:")
            logger.info(f"  Mean daily bias: {np.mean(daily_biases):.3f} TECU")
            logger.info(f"  Std daily bias: {np.std(daily_biases):.3f} TECU")
            logger.info(f"  Min daily bias: {np.min(daily_biases):.3f} TECU")
            logger.info(f"  Max daily bias: {np.max(daily_biases):.3f} TECU")
        
        if stats['processed_observations'] == 0:
            logger.error("No observations were processed. Check model availability and data paths.")
            return
        
        # Save results
        pipeline.save_results(args.output_dir)
        
        # Create plots
        pipeline.plot_results(args.output_dir, args.save_plots)
        
        logger.info(f"Analysis complete! Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()
