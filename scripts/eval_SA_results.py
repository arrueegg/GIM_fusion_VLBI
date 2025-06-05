import os
import yaml
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

import re
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def read_SA_metrics(folder, file_name):
    """
    Parse a metrics.txt with the following structure:

      GLOBAL BIAS: <value>        (ignored here or can be stored separately)

      RAW GLOBAL METRICS:
        RMSE:        <rmse_raw>
        MAE:         <mae_raw>
        STD:         <std_raw>
        Correlation: <corr_raw>

      GLOBAL-BIAS-CORRECTED GLOBAL METRICS:
        RMSE:        <rmse_bcg>
        MAE:         <mae_bcg>
        STD:         <std_bcg>
        Correlation: <corr_bcg>

      PER-STATION METRICS (within <radius> km of station, during station VLBI window):
        Station        Count   RMSE_raw   MAE_raw  RMSE_globCor MAE_globCor  RMSE_locCor  MAE_locCor
        <Station1>       123      3.21      2.45       3.15        2.38         2.80        2.30
        <Station2>       45       4.05      3.10       3.90        3.00         3.80        2.90
         …

    Returns a dictionary:
      {
        'global_rmse_raw':  float,
        'global_mae_raw':   float,
        'global_std_raw':   float,
        'global_corr_raw':  float,
        'global_rmse_bcg':  float,
        'global_mae_bcg':   float,
        'global_std_bcg':   float,
        'global_corr_bcg':  float,
        'station_list':     [ 'station1', 'station2', … ],
        'station_data':     { 
             'station1': { 'count': int,
                           'rmse_raw': float,  'mae_raw': float,
                           'rmse_globCor': float, 'mae_globCor': float,
                           'rmse_locCor': float, 'mae_locCor': float },
             …
        }
      }
    """
    file_path = os.path.join(folder, file_name)
    with open(file_path, 'r') as f:
        lines = [ln.rstrip() for ln in f]

    # Initialize outputs to None or empty
    metrics = {
        'global_rmse_raw': None,
        'global_mae_raw':  None,
        'global_std_raw':  None,
        'global_corr_raw': None,
        'global_rmse_bcg': None,
        'global_mae_bcg':  None,
        'global_std_bcg':  None,
        'global_corr_bcg': None,
    }
    station_data = {}

    # 1) Find the line indices of each section
    raw_idx    = None
    bc_idx     = None
    persta_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("RAW GLOBAL METRICS"):
            raw_idx = i
        elif ln.strip().startswith("GLOBAL-BIAS-CORRECTED GLOBAL METRICS"):
            bc_idx = i
        elif ln.strip().startswith("PER-STATION METRICS"):
            persta_idx = i

    # 2) Parse RAW GLOBAL METRICS (if present)
    if raw_idx is not None:
        # Next few lines typically look like:
        #   RMSE:        5.26
        #   MAE:         3.61
        #   STD:         5.23
        #   Correlation: 0.908
        for offset in range(1,5):
            line = lines[raw_idx + offset].strip()
            if line.startswith("RMSE:"):
                metrics['global_rmse_raw'] = float(line.split(":",1)[1].strip())
            elif line.startswith("MAE:"):
                metrics['global_mae_raw'] = float(line.split(":",1)[1].strip())
            elif line.startswith("STD:"):
                metrics['global_std_raw'] = float(line.split(":",1)[1].strip())
            elif line.startswith("Correlation:"):
                metrics['global_corr_raw'] = float(line.split(":",1)[1].strip())

    # 3) Parse GLOBAL-BIAS-CORRECTED GLOBAL METRICS (if present)
    if bc_idx is not None:
        # Next few lines:
        #   RMSE:        5.23
        #   MAE:         3.51
        #   STD:         5.23
        #   Correlation: 0.908
        for offset in range(1,5):
            line = lines[bc_idx + offset].strip()
            if line.startswith("RMSE:"):
                metrics['global_rmse_bcg'] = float(line.split(":",1)[1].strip())
            elif line.startswith("MAE:"):
                metrics['global_mae_bcg'] = float(line.split(":",1)[1].strip())
            elif line.startswith("STD:"):
                metrics['global_std_bcg'] = float(line.split(":",1)[1].strip())
            elif line.startswith("Correlation:"):
                metrics['global_corr_bcg'] = float(line.split(":",1)[1].strip())

    # 4) Parse PER-STATION table (if present)
    if persta_idx is not None:
        # The header is on persta_idx+1, e.g.
        #   Station        Count   RMSE_raw   MAE_raw  RMSE_globCor MAE_globCor  RMSE_locCor  MAE_locCor
        # Then each station row is fixed-width columns. We can just split by whitespace.
        header_line = lines[persta_idx + 1]
        # Build a list of column names from the header (split on spaces):
        # ["Station", "Count", "RMSE_raw", "MAE_raw", "RMSE_globCor", "MAE_globCor", "RMSE_locCor", "MAE_locCor"]
        cols = header_line.strip().split()
        # Now parse each subsequent line until a blank or end-of-file
        for line in lines[persta_idx + 2:]:
            if not line.strip():
                break
            # Split on whitespace—there should be exactly len(cols) tokens
            tokens = line.strip().split()
            if len(tokens) < len(cols):
                continue  # skip malformed
            st_name = tokens[0]
            vals    = tokens[1:]
            try:
                d = {
                    'count'       : int(vals[0]),
                    'rmse_raw'    : float(vals[1]),
                    'mae_raw'     : float(vals[2]),
                    'rmse_globCor': float(vals[3]),
                    'mae_globCor' : float(vals[4]),
                    'rmse_locCor' : float(vals[5]),
                    'mae_locCor'  : float(vals[6]),
                }
            except ValueError:
                # if some column says "N/A" or similar, wrap with try/except
                d = {
                    'count'       : int(vals[0]) if vals[0].isdigit() else None,
                    'rmse_raw'    : float(vals[1]) if _is_number(vals[1]) else None,
                    'mae_raw'     : float(vals[2]) if _is_number(vals[2]) else None,
                    'rmse_globCor': float(vals[3]) if _is_number(vals[3]) else None,
                    'mae_globCor' : float(vals[4]) if _is_number(vals[4]) else None,
                    'rmse_locCor' : float(vals[5]) if _is_number(vals[5]) else None,
                    'mae_locCor'  : float(vals[6]) if _is_number(vals[6]) else None,
                }
            station_data[st_name.lower()] = d

    # 5) Build final dictionary
    result = metrics.copy()
    result['station_list'] = sorted(station_data.keys())
    result['station_data'] = station_data
    return result, sorted(station_data.keys())


def _is_number(s):
    """Helper: return True if string s can be cast to float."""
    try:
        float(s)
        return True
    except ValueError:
        return False

def read_config_file(folder, file_name):
    file_path = os.path.join(folder, file_name)
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def read_experiment(experiment_folder, experiment_name):
    # Paths
    config_path  = os.path.join(experiment_folder, experiment_name, 'config.yaml')
    metrics_path = os.path.join(experiment_folder, experiment_name, 'SA_plots', 'metrics.txt')

    # 1) Read and flatten config
    cfg = read_config_file(experiment_folder + '/', os.path.join(experiment_name, 'config.yaml'))
    flat = {}
    def _flatten_dict(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict):
                _flatten_dict(v, prefix=f"{prefix}{k}_")
            else:
                flat[f"{prefix}{k}"] = v
    _flatten_dict(cfg)

    # 2) Read new metrics.txt
    SA_results, station_list = read_SA_metrics(
        os.path.join(experiment_folder, experiment_name, 'SA_plots'),
        'metrics.txt'
    )

    # 3) Build data dict
    data = {k: [v] for k, v in flat.items()}

    # 3a) Pick which global metrics to call “global_RMSE” & “global_MAE”
    #    (Here, we choose the RAW ones. If you prefer bias‐corrected, swap the keys.)
    data['global_RMSE'] = [SA_results.get('global_rmse_raw', None)]
    data['global_MAE']  = [SA_results.get('global_mae_raw',  None)]

    # 3b) For each station, bring in raw RMSE/MAE into columns named "{st}_RMSE", "{st}_MAE"
    #    If you also want bias‐corrected or local‐corrected, create additional columns here.
    for st in station_list:
        d = SA_results['station_data'][st]
        data[f"{st}_N"]       = [d['count']]
        data[f"{st}_RMSE"]    = [d['rmse_raw']]
        data[f"{st}_MAE"]     = [d['mae_raw']]
        data[f"{st}_RMSE_bcg"] = [d['rmse_globCor']]
        data[f"{st}_MAE_bcg"]  = [d['mae_globCor']]
        data[f"{st}_RMSE_loc"] = [d['rmse_locCor']]
        data[f"{st}_MAE_loc"]  = [d['mae_locCor']]

    return pd.DataFrame(data)


def evaluate(df):
    # (Most of your original evaluate(...) can stay the same, except:)
    # 1) After reading all experiments into df, you no longer have columns ending in "_RMSE"
    #    for stations—now they’re exactly "{station}_RMSE" and "{station}_MAE". So:
    station_names = sorted([c[:-5]  # remove "_RMSE"
        for c in df.columns if c.endswith('_RMSE') and c != 'global_RMSE'
    ])

    # Your grouping keys and name_map remain the same.
    group_columns = ['data_mode', 'training_vlbi_loss_weight', 'training_vlbi_sampling_weight']
    expected = {
        ('GNSS', 1, 1),
        ('Fusion', 1000, 1),
        ('Fusion', 1, 1000),
        ('DTEC_Fusion', 100, 1),
        ('DTEC_Fusion', 1, 100)
    }
    name_map = {
        ('GNSS', 1, 1):       'GNSS',
        ('Fusion', 1000, 1):  'Fusion LW',
        ('Fusion', 1, 1000):  'Fusion SW',
        ('DTEC_Fusion', 100, 1): 'DTEC LW',
        ('DTEC_Fusion', 1, 100): 'DTEC SW'
    }

    # 1a) Filter to DOYs where all 5 combos are present
    valid_doys = df.groupby('doy').filter(
        lambda g: set(map(tuple, g[group_columns].values)) == expected
    )['doy'].unique()
    df = df[df['doy'].isin(valid_doys)]

    # 2) Convert global to numeric
    df['global_RMSE'] = pd.to_numeric(df['global_RMSE'], errors='coerce')
    df['global_MAE']  = pd.to_numeric(df['global_MAE'],  errors='coerce')

    # 3) Group by the 3‐tuple
    df_mode = df.groupby(group_columns)
    group_keys = list(name_map.keys())
    labels = [name_map[k] for k in group_keys]

    # 4) Global histograms (unchanged)
    bins_rmse = np.linspace(df['global_RMSE'].min(), df['global_RMSE'].max(), 20)
    bins_mae  = np.linspace(df['global_MAE'].min(),  df['global_MAE'].max(),  20)
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
    os.makedirs('evaluation', exist_ok=True)
    plt.savefig('evaluation/RMSE_MAE_hist_by_data_mode.png', dpi=300)
    plt.close()

    # 5) Per‐station histograms (using new station columns)
    for st in station_names:
        bins_st_rmse = np.linspace(df[f'{st}_RMSE'].min(), df[f'{st}_RMSE'].max(), 20)
        bins_st_mae  = np.linspace(df[f'{st}_MAE'].min(),  df[f'{st}_MAE'].max(),  20)

        plt.figure(figsize=(14,6))
        plt.subplot(1,2,1)
        for key in group_keys:
            grp = df_mode.get_group(key)
            plt.hist(grp[f'{st}_RMSE'].dropna(), bins=bins_st_rmse,
                     alpha=0.5, label=name_map[key], edgecolor='black')
        plt.title(f'{st.capitalize()} RMSE (raw) by Data Mode')
        plt.xlabel('RMSE')
        plt.legend()

        plt.subplot(1,2,2)
        for key in group_keys:
            grp = df_mode.get_group(key)
            plt.hist(grp[f'{st}_MAE'].dropna(), bins=bins_st_mae,
                     alpha=0.5, label=name_map[key], edgecolor='black')
        plt.title(f'{st.capitalize()} MAE (raw) by Data Mode')
        plt.xlabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'evaluation/{st}_hist_by_data_mode.png', dpi=300)
        plt.close()

    # 6) Global boxplots (unchanged, since we renamed to `global_RMSE`/`global_MAE`)
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
    plt.savefig('evaluation/RMSE_MAE_boxplots_by_data_mode.png', dpi=300)
    plt.close()

    # 7) Per‐station boxplots (using the new station columns)
    for st in station_names:
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
        plt.savefig(f'evaluation/{st}_box_by_data_mode.png', dpi=300)
        plt.close()

    # 8) Compute medians and reorder
    median_values = df.groupby(group_columns).median(numeric_only=True)
    median_values = median_values.reindex(group_keys)

    global_metrics  = median_values[['global_RMSE', 'global_MAE']].copy()
    station_cols    = [c for c in median_values.columns if any(st in c for st in station_names)]
    station_metrics = median_values[station_cols].copy()

    global_metrics.index  = [name_map[k] for k in global_metrics.index]
    station_metrics.index = [name_map[k] for k in station_metrics.index]

    # 9) Sanity‐check medians
    for i, key in enumerate(group_keys):
        raw_med = np.median(data_rmse[i])
        pd_med  = global_metrics.loc[name_map[key], 'global_RMSE']
        if not np.isclose(raw_med, pd_med, atol=1e-8):
            print(f"WARNING: mismatch in global_RMSE median for {name_map[key]}: "
                  f"{raw_med:.6f} vs {pd_med:.6f}")

        raw_med2 = np.median(data_mae[i])
        pd_med2  = global_metrics.loc[name_map[key], 'global_MAE']
        if not np.isclose(raw_med2, pd_med2, atol=1e-8):
            print(f"WARNING: mismatch in global_MAE median for {name_map[key]}: "
                  f"{raw_med2:.6f} vs {pd_med2:.6f}")

    # 10) Save the CSVs
    os.makedirs('scripts', exist_ok=True)
    global_metrics.to_csv('evaluation/global_median_values.csv', header=True, index_label='Method')
    station_metrics.to_csv('evaluation/station_median_values.csv', header=True, index_label='Method')

    print("Saved global_median_values.csv and station_median_values.csv in 'evaluation/'")


def main():
    experiments_folder = '/cluster/work/igp_psr/arrueegg/WP2/GIM_fusion_VLBI/experiments/'
    df = pd.DataFrame()

    for exp in os.listdir(experiments_folder):
        metrics_txt = os.path.join(experiments_folder, exp, 'SA_plots', 'metrics.txt')
        if os.path.exists(metrics_txt):
            df = pd.concat([df, read_experiment(experiments_folder, exp)], ignore_index=True)

    print("Evaluating results…")
    evaluate(df)


if __name__ == "__main__":
    main()
