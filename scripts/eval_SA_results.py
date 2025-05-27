
import os
import yaml
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
#import seaborn as sns

def read_config_file(folder, file_name):
    file_path = os.path.join(folder, file_name)
    with open(file_path, 'r') as file:
        content = yaml.safe_load(file)
    return content

def read_SA_metrics(folder, file_name):
    """
    Parse a metrics.txt with:
      GLOBAL RMSE: <value>
      GLOBAL MAE: <value>
    PER-STATION METRICS (within <radius> km):
        StationName: RMSE=<v>, MAE=<v>, N=<v>
    Returns dict: 'global_rmse', 'global_mae', '<station>_rmse', '<station>_mae', '<station>_n'.
    """
    file_path = os.path.join(folder, file_name)
    metrics = {}
    with open(file_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # globals
    for ln in lines:
        if ln.startswith('GLOBAL RMSE'):
            metrics['global_rmse'] = float(ln.split(':',1)[1])
        elif ln.startswith('GLOBAL MAE'):
            metrics['global_mae'] = float(ln.split(':',1)[1])

    # per-station
    station_re = re.compile(r"^(\w+):\s*RMSE=([0-9.]+),\s*MAE=([0-9.]+),\s*N=(\d+)")
    stations = []
    for ln in lines:
        m = station_re.match(ln)
        if m:
            name, rmse, mae, n = m.groups()
            key = name.lower()
            stations.append(key)
            metrics[f"{key}_rmse"] = float(rmse)
            metrics[f"{key}_mae"]  = float(mae)
            metrics[f"{key}_n"]    = int(n)
    return metrics, stations

def read_experiment(experiment_folder, experiment_name):

    config_file_path = os.path.join(experiment_name, 'config.yaml')
    SA_results_file_path = os.path.join(experiment_name, 'SA_plots', 'metrics.txt')

    config = read_config_file(experiment_folder, config_file_path)
    SA_results, stations = read_SA_metrics(experiment_folder, SA_results_file_path)

    df_row = pd.DataFrame({f"{key}_{subkey}" if isinstance(subdict, dict) else key: [value] 
                           for key, subdict in config.items() 
                           for subkey, value in (subdict.items() if isinstance(subdict, dict) else [(None, subdict)])})

    df_row["global_RMSE"] = SA_results["global_rmse"]
    df_row["global_MAE"] = SA_results["global_mae"]
    for station in stations:
        df_row[f"{station}_RMSE"] = SA_results.get(f"{station}_rmse", None)
        df_row[f"{station}_MAE"] = SA_results.get(f"{station}_mae", None)
        df_row[f"{station}_N"] = SA_results.get(f"{station}_n", 0)

    return df_row

def evaluate_old(df):
    print("Number of experiments processed in total: ", len(df))
    # Define your grouping columns
    group_columns = ['data_mode', 'training_vlbi_loss_weight', 'training_vlbi_sampling_weight']

    # Expected unique combinations for the groups
    expected_combinations = set([
        ('GNSS', 1, 1),
        ('Fusion', 1000, 1),
        ('Fusion', 1, 1000),
        ('DTEC_Fusion', 100, 1),
        ('DTEC_Fusion', 1, 100)
    ])

    group_name_mapping = {
        ('GNSS', 1, 1): 'GNSS',
        ('Fusion', 1000, 1): 'Fusion LW',
        ('Fusion', 1, 1000): 'Fusion SW',
        ('DTEC_Fusion', 100, 1): 'DTEC LW',
        ('DTEC_Fusion', 1, 100): 'DTEC SW'
    }

    # Group by 'doy' and check if all expected combinations are present
    valid_doys = df.groupby('doy').filter(
        lambda g: set(map(tuple, g[group_columns].values)) == expected_combinations
    )['doy'].unique()

    # Filter original DataFrame to keep only valid DOYs
    df = df[df['doy'].isin(valid_doys)]
    print("Number of experiments where all 5 scenarios were executed successfully: ", len(df))

    # Convert RMSE and MAE to numeric
    df["global RMSE"] = pd.to_numeric(df["global RMSE"], errors='coerce')
    df["global MAE"] = pd.to_numeric(df["global MAE"], errors='coerce')

    df_mode = df.groupby(['data_mode', 'training_vlbi_loss_weight', 'training_vlbi_sampling_weight'])
    
    # Define consistent bins and range for RMSE and MAE
    bins_rmse = np.linspace(df['global RMSE'].min(), df['global RMSE'].max(), 20)
    bins_mae = np.linspace(df['global MAE'].min(), df['global MAE'].max(), 20)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for name, group in df_mode:
        plt.hist(group['global RMSE'], bins=bins_rmse, alpha=0.5, label=group_name_mapping[name], edgecolor='black')
    plt.title('RMSE by Data Mode')
    plt.xlabel('RMSE')
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, group in df_mode:
        plt.hist(group['global MAE'], bins=bins_mae, alpha=0.5, label=group_name_mapping[name], edgecolor='black')
    plt.title('MAE by Data Mode')
    plt.xlabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('scripts/RMSE_MAE_hist_by_data_mode.png', dpi=300)

    # Create subplots with 5 rows and 2 columns (RMSE left, MAE right)
    fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=True, sharey=True)

    for i, (name, group) in enumerate(df_mode):
        mode_label = group_name_mapping.get(name, str(name))
        
        # RMSE subplot (left)
        axes[i, 0].hist(group['global RMSE'], bins=bins_rmse, alpha=0.6, label=f'{mode_label} RMSE', color='tab:blue', edgecolor='black')
        axes[i, 0].set_title(f'{mode_label} RMSE')
        axes[i, 0].set_ylabel('Frequency')
        axes[i, 0].legend()

        # MAE subplot (right)
        axes[i, 1].hist(group['global MAE'], bins=bins_mae, alpha=0.6, label=f'{mode_label} MAE', color='tab:orange', edgecolor='black')
        axes[i, 1].set_title(f'{mode_label} MAE')
        axes[i, 1].legend()

    # Add a common x-axis label for both columns
    axes[-1, 0].set_xlabel('RMSE Value')
    axes[-1, 1].set_xlabel('MAE Value')

    plt.tight_layout()
    plt.savefig('scripts/RMSE_MAE_hist_subplots.png', dpi=300)

    plt.figure(figsize=(12, 6))

    # First subplot (RMSE)
    ax1 = plt.subplot(1, 2, 1)
    data_rmse = [group['global RMSE'].values for name, group in df_mode]
    plt.boxplot(data_rmse, tick_labels=[group_name_mapping[name] for name in df_mode.groups.keys()])
    ax1.set_ylim(bottom=0)
    plt.title('RMSE by Data Mode')
    plt.xlabel('Data Mode')
    plt.ylabel('RMSE')

    # Second subplot (MAE)
    ax2 = plt.subplot(1, 2, 2)
    data_mae = [group['global MAE'].values for name, group in df_mode]
    plt.boxplot(data_mae, tick_labels=[group_name_mapping[name] for name in df_mode.groups.keys()])
    ax2.set_ylim(bottom=0) 
    plt.title('MAE by Data Mode')
    plt.xlabel('Data Mode')
    plt.ylabel('MAE')

    # Save the plot with full range
    plt.tight_layout()
    plt.savefig('scripts/RMSE_MAE_boxplots_by_data_mode.png', dpi=300)

    # Adjust limits for cropped version
    ax1.set_ylim(3, 12) 
    ax2.set_ylim(2, 8) 

    plt.tight_layout()
    plt.savefig('scripts/RMSE_MAE_boxplots_by_data_mode_nooutliers.png', dpi=300)

def evaluate(df):
    print("Number of experiments processed in total:", len(df))
    group_columns = ['data_mode', 'training_vlbi_loss_weight', 'training_vlbi_sampling_weight']
    expected = {('GNSS',1,1),('Fusion',1000,1),('Fusion',1,1000),('DTEC_Fusion',100,1),('DTEC_Fusion',1,100)}
    name_map = {('GNSS',1,1):'GNSS',('Fusion',1000,1):'Fusion LW',('Fusion',1,1000):'Fusion SW',
                ('DTEC_Fusion',100,1):'DTEC LW',('DTEC_Fusion',1,100):'DTEC SW'}
    valid_doys = df.groupby('doy').filter(
        lambda g: set(map(tuple, g[group_columns].values)) == expected
    )['doy'].unique()
    df = df[df['doy'].isin(valid_doys)]
    print("Number of experiments where all 5 scenarios were executed successfully:", len(df))

    # ensure numeric
    df['global_RMSE'] = pd.to_numeric(df['global_RMSE'], errors='coerce')
    df['global_MAE'] = pd.to_numeric(df['global_MAE'], errors='coerce')
    stations = sorted([c[:-5] for c in df.columns if c.endswith('_RMSE') and c!='global_RMSE'])

    df_mode = df.groupby(group_columns)

    # Global histograms
    bins_rmse = np.linspace(df['global_RMSE'].min(), df['global_RMSE'].max(), 20)
    bins_mae = np.linspace(df['global_MAE'].min(), df['global_MAE'].max(), 20)
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    for name, grp in df_mode:
        plt.hist(grp['global_RMSE'], bins=bins_rmse, alpha=0.5, label=name_map[name], edgecolor='black')
    plt.title('Global RMSE by Data Mode'); plt.xlabel('RMSE'); plt.legend()
    plt.subplot(1,2,2)
    for name, grp in df_mode:
        plt.hist(grp['global_MAE'], bins=bins_mae, alpha=0.5, label=name_map[name], edgecolor='black')
    plt.title('Global MAE by Data Mode'); plt.xlabel('MAE'); plt.legend()
    plt.tight_layout(); plt.savefig('scripts/RMSE_MAE_hist_by_data_mode.png', dpi=300); plt.close()

    # Per-station histograms
    for st in stations:
        bins_st_rmse = np.linspace(df[f'{st}_RMSE'].min(), df[f'{st}_RMSE'].max(), 20)
        bins_st_mae  = np.linspace(df[f'{st}_MAE'].min(),  df[f'{st}_MAE'].max(),  20)
        plt.figure(figsize=(14,6))
        plt.subplot(1,2,1)
        for name, grp in df_mode:
            plt.hist(grp[f'{st}_RMSE'], bins=bins_st_rmse, alpha=0.5, label=name_map[name], edgecolor='black')
        plt.title(f'{st.capitalize()} RMSE by Data Mode'); plt.xlabel('RMSE'); plt.legend()
        plt.subplot(1,2,2)
        for name, grp in df_mode:
            plt.hist(grp[f'{st}_MAE'], bins=bins_st_mae, alpha=0.5, label=name_map[name], edgecolor='black')
        plt.title(f'{st.capitalize()} MAE by Data Mode'); plt.xlabel('MAE'); plt.legend()
        plt.tight_layout(); plt.savefig(f'scripts/{st}_hist_by_data_mode.png', dpi=300); plt.close()

    # Global boxplots
    plt.figure(figsize=(12,6))
    data_rmse = [grp['global_RMSE'].values for _,grp in df_mode]
    data_mae  = [grp['global_MAE'].values  for _,grp in df_mode]
    labels = list(name_map.values())
    plt.subplot(1,2,1)
    plt.boxplot(data_rmse, labels=labels); plt.title('Global RMSE by Data Mode'); plt.ylabel('RMSE')
    plt.ylim(3, 12)
    plt.subplot(1,2,2)
    plt.boxplot(data_mae, labels=labels); plt.title('Global MAE by Data Mode'); plt.ylabel('MAE')
    plt.ylim(3, 12) 
    plt.tight_layout(); plt.savefig('scripts/RMSE_MAE_boxplots_by_data_mode.png', dpi=300)

    # Global boxplots no outliers
    plt.subplot(1,2,1); plt.ylim(3,12)
    plt.subplot(1,2,2); plt.ylim(2,8)
    plt.tight_layout(); plt.savefig('scripts/RMSE_MAE_boxplots_by_data_mode_nooutliers.png', dpi=300)

    # Per-station boxplots
    for st in stations:
        data_s_rmse = [grp[f'{st}_RMSE'].values for _,grp in df_mode]
        data_s_mae  = [grp[f'{st}_MAE'].values  for _,grp in df_mode]
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.boxplot(data_s_rmse, labels=labels); plt.title(f'{st.capitalize()} RMSE by Data Mode')
        plt.ylim(3, 12)
        plt.subplot(1,2,2)
        plt.boxplot(data_s_mae, labels=labels); plt.title(f'{st.capitalize()} MAE by Data Mode')
        plt.ylim(3, 12) 
        plt.tight_layout(); plt.savefig(f'scripts/{st}_box_by_data_mode.png', dpi=300)


def main():
    """
    Main function to orchestrate the evaluation process.
    """

    experiments_folder = '/cluster/work/igp_psr/arrueegg/WP2/GIM_fusion_VLBI/experiments/'
    experiments_folder = '/scratch2/arrueegg/WP2/GIM_fusion_VLBI/experiments/'
    
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