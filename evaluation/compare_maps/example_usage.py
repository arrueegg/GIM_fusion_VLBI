#!/usr/bin/env python3
"""
Example usage script for the Jason-3 residual analysis pipeline.

This script demonstrates how to run the pipeline for different approaches
and provides examples of loading and analyzing the results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the evaluation directory to path
sys.path.append(str(Path(__file__).parent))

from plot_jason3_residuals import Jason3ResidualPipeline


def run_analysis_example():
    """Example of running the full analysis pipeline."""
    
    # Change to repository root
    repo_root = Path(__file__).parent.parent.parent
    os.chdir(repo_root)
    
    # Import config after changing directory
    sys.path.append('src')
    from utils.config_parser import parse_config
    
    print("Jason-3 Residual Analysis Example")
    print("=" * 40)
    
    # Load configuration
    config = parse_config()
    
    # Example 1: Analyze GNSS approach
    print("\n1. Analyzing GNSS approach for 2023...")
    try:
        pipeline_gnss = Jason3ResidualPipeline(config, approach="GNSS", year=2023)
        stats_gnss = pipeline_gnss.process_all_observations()
        
        if stats_gnss['processed_observations'] > 0:
            print(f"   ✓ Processed {stats_gnss['processed_observations']} observations")
            
            # Save results
            output_dir = "evaluation/compare_maps/results_gnss"
            pipeline_gnss.save_results(output_dir)
            pipeline_gnss.plot_results(output_dir)
            
            print(f"   ✓ Results saved to {output_dir}")
        else:
            print("   ✗ No observations processed for GNSS")
            
    except Exception as e:
        print(f"   ✗ GNSS analysis failed: {e}")
    
    # Example 2: Analyze Fusion approach (if models available)
    print("\n2. Analyzing Fusion approach for 2023...")
    try:
        pipeline_fusion = Jason3ResidualPipeline(config, approach="Fusion", year=2023)
        stats_fusion = pipeline_fusion.process_all_observations()
        
        if stats_fusion['processed_observations'] > 0:
            print(f"   ✓ Processed {stats_fusion['processed_observations']} observations")
            
            # Save results
            output_dir = "evaluation/compare_maps/results_fusion"
            pipeline_fusion.save_results(output_dir)
            pipeline_fusion.plot_results(output_dir)
            
            print(f"   ✓ Results saved to {output_dir}")
        else:
            print("   ✗ No observations processed for Fusion")
            
    except Exception as e:
        print(f"   ✗ Fusion analysis failed: {e}")


def load_and_compare_results():
    """Example of loading and comparing results from different approaches."""
    
    print("\n3. Comparing Results")
    print("-" * 20)
    
    approaches = ['gnss', 'fusion']
    results = {}
    
    for approach in approaches:
        if approach == 'gnss':
            results_file = f"evaluation/compare_maps/results_{approach}/GNSS_2023_results.npz"
        elif approach == 'fusion':
            results_file = f"evaluation/compare_maps/results_{approach}/Fusion_2023_results.npz"

        if os.path.exists(results_file):
            try:
                data = np.load(results_file)
                results[approach] = data
                
                # Calculate statistics
                residuals = data['residuals']
                valid_residuals = residuals[~np.isnan(residuals)]
                
                if len(valid_residuals) > 0:
                    rmse = np.sqrt(np.mean(valid_residuals**2))
                    mae = np.mean(np.abs(valid_residuals))
                    bias = np.mean(valid_residuals)
                    grid_coverage = np.sum(data['counts'] > 0)
                    
                    print(f"\n{approach.upper()} Results:")
                    print(f"  RMSE: {rmse:.3f} TECU")
                    print(f"  MAE:  {mae:.3f} TECU")
                    print(f"  Bias: {bias:.3f} TECU")
                    print(f"  Grid cells with data: {grid_coverage}")
                    
                else:
                    print(f"\n{approach.upper()}: No valid results found")
                    
            except Exception as e:
                print(f"\n{approach.upper()}: Failed to load results - {e}")
        else:
            print(f"\n{approach.upper()}: Results file not found")
    
    # Create comparison plot if we have results from multiple approaches
    if len(results) > 1:
        create_comparison_plot(results)


def create_comparison_plot(results_dict):
    """Create a comparison plot of residuals from different approaches."""
    
    print("\n4. Creating comparison plot...")
    
    fig, axes = plt.subplots(1, len(results_dict), figsize=(6 * len(results_dict), 5))
    if len(results_dict) == 1:
        axes = [axes]
    
    for i, (approach, data) in enumerate(results_dict.items()):
        residuals = data['residuals']
        valid_residuals = residuals[~np.isnan(residuals)]
        
        if len(valid_residuals) > 0:
            axes[i].hist(valid_residuals, bins=50, alpha=0.7, edgecolor='black')
            axes[i].axvline(0, color='red', linestyle='--', alpha=0.7)
            axes[i].set_title(f'{approach.upper()} Residuals')
            axes[i].set_xlabel('Residuals (TECU)')
            axes[i].set_ylabel('Frequency')
            
            # Add statistics
            rmse = np.sqrt(np.mean(valid_residuals**2))
            mae = np.mean(np.abs(valid_residuals))
            bias = np.mean(valid_residuals)
            
            axes[i].text(0.05, 0.95, 
                        f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nBias: {bias:.2f}',
                        transform=axes[i].transAxes, va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('evaluation/compare_maps/approach_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✓ Comparison plot saved as 'approach_comparison.png'")


def quick_demo():
    """Quick demonstration with minimal processing."""
    
    print("\nQuick Demo Mode")
    print("=" * 15)
    print("This demo processes a small subset of data for demonstration.")
    
    # Change to repository root
    repo_root = Path(__file__).parent.parent.parent
    os.chdir(repo_root)
    
    # Import config after changing directory
    sys.path.append('src')
    from utils.config_parser import parse_config
    
    try:
        config = parse_config()
        pipeline = Jason3ResidualPipeline(config, approach="GNSS", year=2023)
        
        # Load Jason-3 data
        jason3_data = pipeline.load_jason3_data()
        print(f"Loaded {len(jason3_data)} Jason-3 observations for 2023")
        
        # Show data coverage by month
        jason3_data['month'] = jason3_data['time'].dt.month
        monthly_counts = jason3_data.groupby('month').size()
        
        print("\nObservations per month:")
        for month, count in monthly_counts.items():
            print(f"  Month {month:2d}: {count:,} observations")
        
        # Check model availability for first few days
        print("\nChecking model availability (first 10 days):")
        for doy in range(1, 11):
            model_dir = pipeline.get_model_path(doy)
            status = "✓" if model_dir else "✗"
            print(f"  DOY {doy:3d}: {status}")
        
        print("\nDemo complete!")
        
    except Exception as e:
        print(f"Demo failed: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Jason-3 Analysis Examples')
    parser.add_argument('--mode', choices=['full', 'compare', 'demo'], 
                       default='full',
                       help='Analysis mode to run')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_analysis_example()
        load_and_compare_results()
    elif args.mode == 'compare':
        load_and_compare_results()
    elif args.mode == 'demo':
        quick_demo()
    
    print("\nFor full analysis, run:")
    print("python evaluation/compare_maps/example_usage.py --mode full")
