
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

def evaluate(df):
    # ——————————————————————————————————————————————
    # 1) Initial filtering (exactly as before)
    # ——————————————————————————————————————————————
    print("Number of experiments processed in total:", len(df))
    group_columns = ['data_mode', 'training_vlbi_loss_weight', 'training_vlbi_sampling_weight']
    expected = {
        ('GNSS', 1, 1),
        ('Fusion', 1000, 1),
        ('Fusion', 1, 1000),
        ('DTEC_Fusion', 100, 1),
        ('DTEC_Fusion', 1, 100)
    }
    name_map = {
        ('GNSS', 1, 1): 'GNSS',
        ('Fusion', 1000, 1): 'Fusion LW',
        ('Fusion', 1, 1000): 'Fusion SW',
        ('DTEC_Fusion', 100, 1): 'DTEC LW',
        ('DTEC_Fusion', 1, 100): 'DTEC SW'
    }

    # 1a) Keep only DOYs where all 5 group combos are present
    valid_doys = df.groupby('doy').filter(
        lambda g: set(map(tuple, g[group_columns].values)) == expected
    )['doy'].unique()

    df = df[df['doy'].isin(valid_doys)]
    print("Number of experiments where all 5 scenarios were executed successfully:", len(df))

    # ——————————————————————————————————————————————
    # 2) Force numeric columns and find station list
    # ——————————————————————————————————————————————
    df['global_RMSE'] = pd.to_numeric(df['global_RMSE'], errors='coerce')
    df['global_MAE'] = pd.to_numeric(df['global_MAE'], errors='coerce')

    # All columns that end with "_RMSE" except "global_RMSE"
    stations = sorted(
        [c[:-5] for c in df.columns if c.endswith('_RMSE') and c != 'global_RMSE']
    )

    # ——————————————————————————————————————————————
    # 3) Group the DataFrame once (but we will pull groups explicitly in a fixed order)
    # ——————————————————————————————————————————————
    df_mode = df.groupby(group_columns)

    # Define a fixed, explicit order for the groups
    group_keys = list(name_map.keys())
    labels = [name_map[k] for k in group_keys]

    # ——————————————————————————————————————————————
    # 4) Global histograms (unchanged)
    # ——————————————————————————————————————————————
    bins_rmse = np.linspace(df['global_RMSE'].min(), df['global_RMSE'].max(), 20)
    bins_mae = np.linspace(df['global_MAE'].min(), df['global_MAE'].max(), 20)
    plt.figure(figsize=(14,6))

    plt.subplot(1,2,1)
    for key in group_keys:
        grp = df_mode.get_group(key)
        plt.hist(grp['global_RMSE'].dropna(), bins=bins_rmse,
                 alpha=0.5, label=name_map[key], edgecolor='black')
    plt.title('Global RMSE by Data Mode')
    plt.xlabel('RMSE')
    plt.legend()

    plt.subplot(1,2,2)
    for key in group_keys:
        grp = df_mode.get_group(key)
        plt.hist(grp['global_MAE'].dropna(), bins=bins_mae,
                 alpha=0.5, label=name_map[key], edgecolor='black')
    plt.title('Global MAE by Data Mode')
    plt.xlabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('scripts/RMSE_MAE_hist_by_data_mode.png', dpi=300)
    plt.close()

    # ——————————————————————————————————————————————
    # 5) Per‐station histograms (unchanged, but dropna())
    # ——————————————————————————————————————————————
    for st in stations:
        bins_st_rmse = np.linspace(df[f'{st}_RMSE'].min(), df[f'{st}_RMSE'].max(), 20)
        bins_st_mae  = np.linspace(df[f'{st}_MAE'].min(),  df[f'{st}_MAE'].max(),  20)

        plt.figure(figsize=(14,6))
        plt.subplot(1,2,1)
        for key in group_keys:
            grp = df_mode.get_group(key)
            plt.hist(grp[f'{st}_RMSE'].dropna(), bins=bins_st_rmse,
                     alpha=0.5, label=name_map[key], edgecolor='black')
        plt.title(f'{st.capitalize()} RMSE by Data Mode')
        plt.xlabel('RMSE')
        plt.legend()

        plt.subplot(1,2,2)
        for key in group_keys:
            grp = df_mode.get_group(key)
            plt.hist(grp[f'{st}_MAE'].dropna(), bins=bins_st_mae,
                     alpha=0.5, label=name_map[key], edgecolor='black')
        plt.title(f'{st.capitalize()} MAE by Data Mode')
        plt.xlabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'scripts/{st}_hist_by_data_mode.png', dpi=300)
        plt.close()

    # ——————————————————————————————————————————————
    # 6) Global boxplots (using the same raw arrays, dropna)
    # ——————————————————————————————————————————————
    data_rmse = []
    data_mae  = []
    for key in group_keys:
        grp = df_mode.get_group(key)
        data_rmse.append(grp['global_RMSE'].dropna().values)
        data_mae.append(grp['global_MAE'].dropna().values)

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.boxplot(data_rmse, labels=labels)
    plt.title('Global RMSE by Data Mode')
    plt.ylabel('RMSE')
    plt.ylim(3, 12)

    plt.subplot(1,2,2)
    plt.boxplot(data_mae, labels=labels)
    plt.title('Global MAE by Data Mode')
    plt.ylabel('MAE')
    plt.ylim(3, 12)

    plt.tight_layout()
    plt.savefig('scripts/RMSE_MAE_boxplots_by_data_mode.png', dpi=300)
    # (No change for the “no outliers” plot aside from consistent data)
    plt.subplot(1,2,1); plt.ylim(3,12)
    plt.subplot(1,2,2); plt.ylim(2,8)
    plt.tight_layout()
    plt.savefig('scripts/RMSE_MAE_boxplots_by_data_mode_nooutliers.png', dpi=300)
    plt.close()

    # ——————————————————————————————————————————————
    # 7) Per‐station boxplots (again, dropna())
    # ——————————————————————————————————————————————
    for st in stations:
        print(f"Plotting boxplots for station {st}...")
        data_s_rmse = []
        data_s_mae  = []
        for key in group_keys:
            grp = df_mode.get_group(key)
            data_s_rmse.append(grp[f'{st}_RMSE'].dropna().values)
            data_s_mae.append(grp[f'{st}_MAE'].dropna().values)

        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.boxplot(data_s_rmse, labels=labels)
        plt.title(f'{st.capitalize()} RMSE by Data Mode')
        plt.ylim(0, 15)

        plt.subplot(1,2,2)
        plt.boxplot(data_s_mae, labels=labels)
        plt.title(f'{st.capitalize()} MAE by Data Mode')
        plt.ylim(0, 12)

        plt.tight_layout()
        plt.savefig(f'scripts/{st}_box_by_data_mode.png', dpi=300)
        plt.close()

    # ——————————————————————————————————————————————
    # 8) Compute groupby‐medians and reorder rows exactly as group_keys
    # ——————————————————————————————————————————————
    median_values = df.groupby(group_columns).median(numeric_only=True)

    # Reindex so the rows come in the same order as our group_keys list
    median_values = median_values.reindex(group_keys)

    # Extract just the columns we need
    global_metrics = median_values[['global_RMSE', 'global_MAE']].copy()
    station_cols = [c for c in median_values.columns if any(st in c for st in stations)]
    station_metrics = median_values[station_cols].copy()

    # Now rename the index from a tuple (e.g. ('GNSS',1,1)) → human label ("GNSS", etc.)
    global_metrics.index = [name_map[k] for k in global_metrics.index]
    station_metrics.index = [name_map[k] for k in station_metrics.index]

    # ——————————————————————————————————————————————
    # 9) Sanity‐check: compare each group’s raw‐array median vs. the Pandas median
    # ——————————————————————————————————————————————
    #    If anything does NOT match, print a warning.
    for i, key in enumerate(group_keys):
        # Recompute median from the raw arrays we used for boxplot
        raw_median_rmse = np.median(data_rmse[i])
        pd_median_rmse  = global_metrics.loc[name_map[key], 'global_RMSE']

        if not np.isclose(raw_median_rmse, pd_median_rmse, atol=1e-8):
            print(f"WARNING: mismatch in 'global_RMSE' median for {name_map[key]}: "
                  f"boxplot‐array median={raw_median_rmse:.6f} vs Pandas median={pd_median_rmse:.6f}")

        raw_median_mae = np.median(data_mae[i])
        pd_median_mae  = global_metrics.loc[name_map[key], 'global_MAE']

        if not np.isclose(raw_median_mae, pd_median_mae, atol=1e-8):
            print(f"WARNING: mismatch in 'global_MAE' median for {name_map[key]}: "
                  f"boxplot‐array median={raw_median_mae:.6f} vs Pandas median={pd_median_mae:.6f}")

    # ——————————————————————————————————————————————
    # 10) Save CSVs (now guaranteed to line up with the boxplot’s order & values)
    # ——————————————————————————————————————————————
    global_metrics.to_csv('scripts/global_median_values.csv',
                          header=True,
                          index_label='Method')

    station_metrics.to_csv('scripts/station_median_values.csv',
                           header=True,
                           index_label='Method')

    print("Saved global_median_values.csv and station_median_values.csv in 'scripts/'")

def main():
    """
    Main function to orchestrate the evaluation process.
    """

    experiments_folder = '/cluster/work/igp_psr/arrueegg/WP2/GIM_fusion_VLBI/experiments/'
    #experiments_folder = '/scratch2/arrueegg/WP2/GIM_fusion_VLBI/experiments/'
    
    df = pd.DataFrame()
    for experiment_name in os.listdir(experiments_folder):
        if os.path.exists(os.path.join(experiments_folder, experiment_name, 'SA_plots', 'metrics.txt')):
            df = pd.concat([df, read_experiment(experiments_folder, experiment_name)], ignore_index=True)

    # Evaluate the results
    print("Evaluating results...")
    evaluation_results = evaluate(df)
    
    # Print or save the evaluation results
    print(evaluation_results)

if __name__ == "__main__":
    main()