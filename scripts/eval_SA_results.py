
import os
import yaml
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#import seaborn as sns

def read_config_file(folder, file_name):
    file_path = os.path.join(folder, file_name)
    with open(file_path, 'r') as file:
        content = yaml.safe_load(file)
    return content

def read_SA_csv(folder, file_name):
    file_path = os.path.join(folder, file_name)
    with open(file_path, 'r') as file:
        content = file.readlines()
    return content

def read_experiment(experiment_folder, experiment_name):

    config_file_path = os.path.join(experiment_name, 'config.yaml')
    SA_results_file_path = os.path.join(experiment_name, 'SA_plots', 'metrics.txt')

    config = read_config_file(experiment_folder, config_file_path)
    SA_results = read_SA_csv(experiment_folder, SA_results_file_path)
    SA_results = [line.split(':') for line in SA_results]

    df_row = pd.DataFrame({f"{key}_{subkey}" if isinstance(subdict, dict) else key: [value] 
                           for key, subdict in config.items() 
                           for subkey, value in (subdict.items() if isinstance(subdict, dict) else [(None, subdict)])})

    df_row["RMSE"] = SA_results[0][1]
    df_row["MAE"] = SA_results[1][1]

    return df_row

def evaluate(df):
    # Convert RMSE and MAE to numeric
    df["RMSE"] = pd.to_numeric(df["RMSE"], errors='coerce')
    df["MAE"] = pd.to_numeric(df["MAE"], errors='coerce')

    # Plot RMSE and MAE for all entries
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.hist(df["RMSE"])
    plt.title('Distribution of RMSE')
    plt.xlabel('RMSE')

    plt.subplot(1, 2, 2)
    plt.hist(df["MAE"])
    plt.title('Distribution of MAE')
    plt.xlabel('MAE')

    plt.tight_layout()
    plt.savefig('scripts/Histograms.png')

    # Group by specific arguments like data_mode and plot
    if 'data_mode' in df.columns:
        df_mode = df.groupby('data_mode')
        plt.figure(figsize=(14, 6))

        # Define consistent bins and range for RMSE
        rmse_min, rmse_max = df['RMSE'].min(), df['RMSE'].max()
        mae_min, mae_max = df['MAE'].min(), df['MAE'].max()
        
        bins_rmse = np.linspace(rmse_min, rmse_max, 20)  # 30 bins for RMSE
        bins_mae = np.linspace(mae_min, mae_max, 20)     # 30 bins for MAE

        plt.subplot(1, 2, 1)
        for name, group in df_mode:
            plt.hist(group['RMSE'], bins=bins_rmse, alpha=0.5, label=name, edgecolor='black')
        plt.title('RMSE by Data Mode')
        plt.xlabel('RMSE')
        plt.legend()

        plt.subplot(1, 2, 2)
        for name, group in df_mode:
            plt.hist(group['MAE'], bins=bins_mae, alpha=0.5, label=name, edgecolor='black')
        plt.title('MAE by Data Mode')
        plt.xlabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.savefig('scripts/RMSE_MAE_hist_by_data_mode.png')

        plt.figure(figsize=(12,6))

        plt.subplot(1, 2, 1)
        data = [group['RMSE'].values for name, group in df_mode]
        plt.boxplot(data, labels=df_mode.groups.keys())
        plt.ylim(bottom=0)
        plt.title('RMSE by Data Mode')
        plt.xlabel('Data Mode')
        plt.ylabel('RMSE')

        plt.subplot(1, 2, 2)
        data = [group['MAE'].values for name, group in df_mode]
        plt.boxplot(data, labels=df_mode.groups.keys())
        plt.ylim(bottom=0)
        plt.title('MAE by Data Mode')
        plt.xlabel('Data Mode')
        plt.ylabel('MAE')

        plt.tight_layout()
        plt.savefig('scripts/RMSE_MAE_boxplots_by_data_mode.png')

def main():
    """
    Main function to orchestrate the evaluation process.
    """

    experiments_folder = '/cluster/work/igp_psr/arrueegg/WP2/GIM_fusion_VLBI/experiments/'
    
    df = pd.DataFrame()
    for experiment_name in os.listdir(experiments_folder):
        if os.path.exists(os.path.join(experiments_folder, experiment_name, 'SA_plots', 'metrics.txt')):
            df = pd.concat([df, read_experiment(experiments_folder, experiment_name)], ignore_index=True)

    # Evaluate the results
    evaluation_results = evaluate(df)
    
    # Print or save the evaluation results
    print(evaluation_results)

if __name__ == "__main__":
    main()