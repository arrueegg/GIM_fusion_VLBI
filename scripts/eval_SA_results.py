
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

def filter_region(df, center_lat, center_lon, radius_deg):
    return df[
        (df.latitude  >= center_lat - radius_deg) &
        (df.latitude  <= center_lat + radius_deg) &
        (df.longitude >= center_lon - radius_deg) &
        (df.longitude <= center_lon + radius_deg)
    ]

def evaluate(df):
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
    df["RMSE"] = pd.to_numeric(df["RMSE"], errors='coerce')
    df["MAE"] = pd.to_numeric(df["MAE"], errors='coerce')

    df_mode = df.groupby(['data_mode', 'training_vlbi_loss_weight', 'training_vlbi_sampling_weight'])
    
    # Define consistent bins and range for RMSE and MAE
    bins_rmse = np.linspace(df['RMSE'].min(), df['RMSE'].max(), 20)
    bins_mae = np.linspace(df['MAE'].min(), df['MAE'].max(), 20)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for name, group in df_mode:
        plt.hist(group['RMSE'], bins=bins_rmse, alpha=0.5, label=group_name_mapping[name], edgecolor='black')
    plt.title('RMSE by Data Mode')
    plt.xlabel('RMSE')
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, group in df_mode:
        plt.hist(group['MAE'], bins=bins_mae, alpha=0.5, label=group_name_mapping[name], edgecolor='black')
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
        axes[i, 0].hist(group['RMSE'], bins=bins_rmse, alpha=0.6, label=f'{mode_label} RMSE', color='tab:blue', edgecolor='black')
        axes[i, 0].set_title(f'{mode_label} RMSE')
        axes[i, 0].set_ylabel('Frequency')
        axes[i, 0].legend()

        # MAE subplot (right)
        axes[i, 1].hist(group['MAE'], bins=bins_mae, alpha=0.6, label=f'{mode_label} MAE', color='tab:orange', edgecolor='black')
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
    data_rmse = [group['RMSE'].values for name, group in df_mode]
    plt.boxplot(data_rmse, tick_labels=[group_name_mapping[name] for name in df_mode.groups.keys()])
    ax1.set_ylim(bottom=0)
    plt.title('RMSE by Data Mode')
    plt.xlabel('Data Mode')
    plt.ylabel('RMSE')

    # Second subplot (MAE)
    ax2 = plt.subplot(1, 2, 2)
    data_mae = [group['MAE'].values for name, group in df_mode]
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

def main():
    """
    Main function to orchestrate the evaluation process.
    """

    STATION_COORDS = {
        'Kokee': (22.126, -159.665),
        # add more stations if needed
    }

    # 2) Region constants
    KOKEE_LAT, KOKEE_LON = STATION_COORDS['Kokee']
    BBOX_RADIUS_DEG = 1.0  # half‐width of your bounding‐box in degrees

    experiments_folder = '/cluster/work/igp_psr/arrueegg/WP2/GIM_fusion_VLBI/experiments/'
    
    df = pd.DataFrame()
    for experiment_name in os.listdir(experiments_folder):
        if os.path.exists(os.path.join(experiments_folder, experiment_name, 'SA_plots', 'metrics.txt')):
            df = pd.concat([df, read_experiment(experiments_folder, experiment_name)], ignore_index=True)

    # Evaluate the results
    evaluation_results = evaluate(df)
    
    # Print or save the evaluation results
    print(evaluation_results)

    df_kokee = filter_region(df, KOKEE_LAT, KOKEE_LON, BBOX_RADIUS_DEG)
    # Evaluate only that subset
    eval_sta_results = evaluate(df_kokee)

    # Print or save the evaluation results for Kokee
    print("Evaluation results for Kokee station:")
    print(eval_sta_results)

if __name__ == "__main__":
    main()