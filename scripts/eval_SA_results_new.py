import os
import yaml
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def read_SA_metrics(folder, file_name):
    file_path = os.path.join(folder, file_name)
    with open(file_path, 'r') as f:
        lines = [ln.rstrip() for ln in f]

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

    raw_idx = bc_idx = persta_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("RAW GLOBAL METRICS"):
            raw_idx = i
        elif ln.strip().startswith("GLOBAL-BIAS-CORRECTED GLOBAL METRICS"):
            bc_idx = i
        elif ln.strip().startswith("PER-STATION METRICS"):
            persta_idx = i

    if raw_idx is not None:
        for offset in range(1, 5):
            line = lines[raw_idx + offset].strip()
            if line.startswith("RMSE:"):
                metrics['global_rmse_raw'] = float(line.split(":",1)[1].strip())
            elif line.startswith("MAE:"):
                metrics['global_mae_raw'] = float(line.split(":",1)[1].strip())
            elif line.startswith("STD:"):
                metrics['global_std_raw'] = float(line.split(":",1)[1].strip())
            elif line.startswith("Correlation:"):
                metrics['global_corr_raw'] = float(line.split(":",1)[1].strip())

    if bc_idx is not None:
        for offset in range(1, 5):
            line = lines[bc_idx + offset].strip()
            if line.startswith("RMSE:"):
                metrics['global_rmse_bcg'] = float(line.split(":",1)[1].strip())
            elif line.startswith("MAE:"):
                metrics['global_mae_bcg'] = float(line.split(":",1)[1].strip())
            elif line.startswith("STD:"):
                metrics['global_std_bcg'] = float(line.split(":",1)[1].strip())
            elif line.startswith("Correlation:"):
                metrics['global_corr_bcg'] = float(line.split(":",1)[1].strip())

    if persta_idx is not None:
        header_line = lines[persta_idx + 1]
        cols = header_line.strip().split()
        for line in lines[persta_idx + 2:]:
            if not line.strip():
                break
            tokens = line.strip().split()
            if len(tokens) < len(cols):
                continue
            st_name = tokens[0]
            vals    = tokens[1:]
            try:
                d = {
                    'count': int(vals[0]),
                    'rmse_raw': float(vals[1]), 'mae_raw': float(vals[2]),
                    'rmse_globCor': float(vals[3]), 'mae_globCor': float(vals[4]),
                    'rmse_locCor': float(vals[5]), 'mae_locCor': float(vals[6]),
                }
            except ValueError:
                d = {
                    'count': int(vals[0]) if vals[0].isdigit() else None,
                    'rmse_raw': float(vals[1]) if _is_number(vals[1]) else None,
                    'mae_raw': float(vals[2]) if _is_number(vals[2]) else None,
                    'rmse_globCor': float(vals[3]) if _is_number(vals[3]) else None,
                    'mae_globCor': float(vals[4]) if _is_number(vals[4]) else None,
                    'rmse_locCor': float(vals[5]) if _is_number(vals[5]) else None,
                    'mae_locCor': float(vals[6]) if _is_number(vals[6]) else None,
                }
            station_data[st_name.lower()] = d

    result = metrics.copy()
    result['station_list'] = sorted(station_data.keys())
    result['station_data'] = station_data
    return result, sorted(station_data.keys())

def read_config_file(folder, file_name):
    file_path = os.path.join(folder, file_name)
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def read_experiment(experiment_folder, experiment_name):
    config_path  = os.path.join(experiment_folder, experiment_name, 'config.yaml')
    metrics_path = os.path.join(experiment_folder, experiment_name, 'SA_plots', 'metrics.txt')

    cfg = read_config_file(experiment_folder + '/', os.path.join(experiment_name, 'config.yaml'))
    flat = {}
    def _flatten_dict(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict):
                _flatten_dict(v, prefix=f"{prefix}{k}_")
            else:
                flat[f"{prefix}{k}"] = v
    _flatten_dict(cfg)

    SA_results, station_list = read_SA_metrics(
        os.path.join(experiment_folder, experiment_name, 'SA_plots'), 'metrics.txt'
    )

    data = {k: [v] for k, v in flat.items()}

    data['global_RMSE'] = [SA_results.get('global_rmse_raw', None)]
    data['global_MAE']  = [SA_results.get('global_mae_raw',  None)]
    data['global_RMSE_globalbias'] = [SA_results.get('global_rmse_bcg', None)]
    data['global_MAE_globalbias']  = [SA_results.get('global_mae_bcg',  None)]

    for st in station_list:
        d = SA_results['station_data'][st]
        data[f"{st}_N"]             = [d['count']]
        data[f"{st}_RMSE"]          = [d['rmse_raw']]
        data[f"{st}_MAE"]           = [d['mae_raw']]
        data[f"{st}_RMSE_bcg"]      = [d['rmse_globCor']]
        data[f"{st}_MAE_bcg"]       = [d['mae_globCor']]
        data[f"{st}_RMSE_loc"]      = [d['rmse_locCor']]
        data[f"{st}_MAE_loc"]       = [d['mae_locCor']]

    return pd.DataFrame(data)

def evaluate(df):
    station_names = sorted([c[:-5]
        for c in df.columns if c.endswith('_RMSE') and c not in ['global_RMSE', 'global_RMSE_bcg']
    ])

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

    # Filter to valid DOYs where all expected group-keys are present
    valid_doys = df.groupby('doy').filter(
        lambda g: set(map(tuple, g[group_columns].values)) == expected
    )['doy'].unique()
    df = df[df['doy'].isin(valid_doys)].copy()

    # Convert global columns to numeric
    df['global_RMSE'] = pd.to_numeric(df['global_RMSE'], errors='coerce')
    df['global_MAE']  = pd.to_numeric(df['global_MAE'],  errors='coerce')
    df['global_RMSE_globalbias'] = pd.to_numeric(df['global_RMSE_globalbias'], errors='coerce')
    df['global_MAE_globalbias']  = pd.to_numeric(df['global_MAE_globalbias'],  errors='coerce')

    # Group by method keys
    df_mode = df.groupby(group_columns)
    group_keys = list(name_map.keys())
    labels = [name_map[k] for k in group_keys]

    os.makedirs('evaluation', exist_ok=True)

    # ---------- Global Histograms (raw & bias-corrected) ----------
    for suffix, title_suffix in [('', 'raw'), ('_globalbias', 'global bias-corrected')]:
        plt.figure(figsize=(14,6))
        rmse_col = f'global_RMSE{suffix}'
        mae_col  = f'global_MAE{suffix}'
        rmse_vals = df[rmse_col].dropna()
        mae_vals  = df[mae_col].dropna()
        if len(rmse_vals) == 0 or len(mae_vals) == 0:
            plt.close()
            continue
        bins_rmse = np.linspace(rmse_vals.min(), rmse_vals.max(), 20)
        bins_mae  = np.linspace(mae_vals.min(),  mae_vals.max(),  20)

        plt.subplot(1,2,1)
        for key in group_keys:
            if key not in df_mode.groups:
                continue
            grp = df_mode.get_group(key)
            vals = grp[rmse_col].dropna()
            if len(vals) > 0:
                plt.hist(vals, bins=bins_rmse, alpha=0.5, label=name_map[key], edgecolor='black')
        plt.title(f'Global RMSE ({title_suffix}) by Data Mode')
        plt.xlabel('RMSE')
        plt.legend()

        plt.subplot(1,2,2)
        for key in group_keys:
            if key not in df_mode.groups:
                continue
            grp = df_mode.get_group(key)
            vals = grp[mae_col].dropna()
            if len(vals) > 0:
                plt.hist(vals, bins=bins_mae, alpha=0.5, label=name_map[key], edgecolor='black')
        plt.title(f'Global MAE ({title_suffix}) by Data Mode')
        plt.xlabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'evaluation/global_hist_{title_suffix.replace(" ", "_")}.png', dpi=300)
        plt.close()

    # ---------- Global Boxplots (raw & bias-corrected) ----------
    for suffix, title_suffix in [('', 'raw'), ('_globalbias', 'global bias-corrected')]:
        rmse_col = f'global_RMSE{suffix}'
        mae_col  = f'global_MAE{suffix}'
        data_rmse = []
        data_mae  = []
        for key in group_keys:
            if key not in df_mode.groups:
                data_rmse.append([])
                data_mae.append([])
                continue
            grp = df_mode.get_group(key)
            data_rmse.append(grp[rmse_col].dropna().values)
            data_mae.append(grp[mae_col].dropna().values)
        if all(len(x) == 0 for x in data_rmse) or all(len(x) == 0 for x in data_mae):
            continue

        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.boxplot(data_rmse, labels=labels)
        plt.title(f'Global RMSE ({title_suffix}) by Data Mode')

        plt.subplot(1,2,2)
        plt.boxplot(data_mae, labels=labels)
        plt.title(f'Global MAE ({title_suffix}) by Data Mode')

        plt.tight_layout()
        plt.savefig(f'evaluation/global_box_{title_suffix.replace(" ", "_")}.png', dpi=300)
        plt.close()

    # ---------- Per-Station Plots (only include rows where count >= 50) ----------
    for st in station_names:
        count_col = f'{st}_N'
        if count_col not in df.columns:
            continue
        mask_valid = df[count_col] >= 50
        df_st = df[mask_valid]
        if df_st.empty:
            continue
        df_mode_st = df_st.groupby(group_columns)

        for col_suffix, title_suffix in [('', 'raw'), ('_bcg', 'global bias-corrected'), ('_loc', 'local bias-corrected')]:
            rmse_col = f'{st}_RMSE{col_suffix}'
            mae_col  = f'{st}_MAE{col_suffix}'
            if rmse_col not in df_st.columns or mae_col not in df_st.columns:
                continue

            rmse_vals = df_st[rmse_col].dropna()
            mae_vals  = df_st[mae_col].dropna()
            if len(rmse_vals) == 0 or len(mae_vals) == 0:
                continue

            bins_st_rmse = np.linspace(rmse_vals.min(), rmse_vals.max(), 20)
            bins_st_mae  = np.linspace(mae_vals.min(),  mae_vals.max(),  20)

            plt.figure(figsize=(14,6))
            plt.subplot(1,2,1)
            for key in group_keys:
                if key not in df_mode_st.groups:
                    continue
                grp = df_mode_st.get_group(key)
                vals = grp[rmse_col].dropna()
                if len(vals) > 0:
                    plt.hist(vals, bins=bins_st_rmse, alpha=0.5, label=name_map[key], edgecolor='black')
            plt.title(f'{st.capitalize()} RMSE ({title_suffix}) by Data Mode')
            plt.xlabel('RMSE')
            plt.legend()

            plt.subplot(1,2,2)
            for key in group_keys:
                if key not in df_mode_st.groups:
                    continue
                grp = df_mode_st.get_group(key)
                vals = grp[mae_col].dropna()
                if len(vals) > 0:
                    plt.hist(vals, bins=bins_st_mae, alpha=0.5, label=name_map[key], edgecolor='black')
            plt.title(f'{st.capitalize()} MAE ({title_suffix}) by Data Mode')
            plt.xlabel('MAE')
            plt.legend()

            plt.tight_layout()
            fname = f'evaluation/{st}_hist_{title_suffix.replace(" ", "_")}.png'
            plt.savefig(fname, dpi=300)
            plt.close()

            # Boxplots
            data_s_rmse = []
            data_s_mae  = []
            for key in group_keys:
                if key not in df_mode_st.groups:
                    data_s_rmse.append([])
                    data_s_mae.append([])
                    continue
                grp = df_mode_st.get_group(key)
                data_s_rmse.append(grp[rmse_col].dropna().values)
                data_s_mae.append(grp[mae_col].dropna().values)
            if all(len(x)==0 for x in data_s_rmse) or all(len(x)==0 for x in data_s_mae):
                continue

            plt.figure(figsize=(12,6))
            plt.subplot(1,2,1)
            plt.boxplot(data_s_rmse, labels=labels)
            plt.title(f'{st.capitalize()} RMSE ({title_suffix}) by Data Mode')

            plt.subplot(1,2,2)
            plt.boxplot(data_s_mae, labels=labels)
            plt.title(f'{st.capitalize()} MAE ({title_suffix}) by Data Mode')

            plt.tight_layout()
            fname = f'evaluation/{st}_box_{title_suffix.replace(" ", "_")}.png'
            plt.savefig(fname, dpi=300)
            plt.close()

    # ---------- Aggregated (all-DOY) Metrics (per-station) ----------
    agg_records = []
    for st in station_names:
        count_col = f"{st}_N"
        if count_col not in df.columns:
            continue

        for key in group_keys:
            if key not in df_mode.groups:
                continue

            df_mode_st = df[df[list(group_columns)].apply(tuple, axis=1) == key]
            if df_mode_st.empty:
                continue

            df_sm = df_mode_st[df_mode_st[count_col] >= 50]
            if df_sm.empty:
                continue

            method_name = name_map[key]
            for suffix, label_suffix in [("", "raw"), ("_bcg", "glob-bias-corr"), ("_loc", "local-bias-corr")]:
                rmse_col = f"{st}_RMSE{suffix}"
                mae_col  = f"{st}_MAE{suffix}"
                if rmse_col not in df_sm.columns or mae_col not in df_sm.columns:
                    continue

                sub = df_sm[[count_col, rmse_col, mae_col]].dropna()
                if sub.empty:
                    continue

                N_arr    = sub[count_col].to_numpy()
                rmse_arr = sub[rmse_col].to_numpy()
                mae_arr  = sub[mae_col].to_numpy()

                weighted_rmse_sq = np.sum(N_arr * (rmse_arr ** 2))
                total_N = np.sum(N_arr)
                if total_N > 0:
                    RMSE_total = np.sqrt(weighted_rmse_sq / total_N)
                    MAE_total  = np.sum(N_arr * mae_arr) / total_N
                else:
                    RMSE_total = np.nan
                    MAE_total  = np.nan

                agg_records.append({
                    "station": st,
                    "method": method_name,
                    "correction": label_suffix,
                    "RMSE_allDOY": RMSE_total,
                    "MAE_allDOY":  MAE_total,
                    "count": total_N,
                })

    if agg_records:
        df_agg = pd.DataFrame(agg_records)
        df_agg = df_agg.sort_values(["station", "method", "correction"])
        df_agg.to_csv("evaluation/station_aggregated_metrics.csv", index=False)
        print("Saved station_aggregated_metrics.csv in 'evaluation/'")

        # ---------- Visualization Section ----------
        os.makedirs('evaluation/visualizations', exist_ok=True)
        df_agg = pd.read_csv("evaluation/station_aggregated_metrics.csv")

        # 1. Bar Plot: RMSE/MAE per Method for a Few Selected Stations
        #selected_stations = ['kokee', 'hobart12', 'ishioka', 'onsala60', 'wark12m',
        #                      'wettzell', 'matera', 'fortleza', 'sejong', 'kokee12m', 'onsa13ne', 'onsa13sw']

        selected_stations = df_agg['station'].unique()
        
        for st in selected_stations:
            df_st = df_agg[df_agg['station'] == st]
            methods = df_st['method'].unique()
            corrections = ['raw', 'glob-bias-corr', 'local-bias-corr']

            x = np.arange(len(methods))
            width = 0.12  # Narrower width to fit all bars
            total_bars = len(corrections) * 2  # 2 metrics per correction

            plt.figure(figsize=(12, 6))

            for i, corr in enumerate(corrections):
                df_corr = df_st[df_st['correction'] == corr]

                rmse_vals = [
                    df_corr[df_corr['method'] == m]['RMSE_allDOY'].values[0]
                    if not df_corr[df_corr['method'] == m].empty
                    else np.nan
                    for m in methods
                ]
                mae_vals = [
                    df_corr[df_corr['method'] == m]['MAE_allDOY'].values[0]
                    if not df_corr[df_corr['method'] == m].empty
                    else np.nan
                    for m in methods
                ]

                offset = (i * 2)  # 2 bars per correction
                plt.bar(x + (offset - total_bars / 2) * width, rmse_vals, width, label=f'RMSE ({corr})', alpha=0.7)
                plt.bar(x + (offset + 1 - total_bars / 2) * width, mae_vals, width, label=f'MAE ({corr})', alpha=0.7)


            plt.xticks(x, methods)
            plt.xlabel('Method')
            plt.ylabel('Error')
            plt.title(f'{st.capitalize()}: RMSE and MAE by Method & Correction')
            plt.legend(loc='upper right', fontsize='small')
            plt.tight_layout()
            plt.savefig(f'evaluation/visualizations/{st}_barplot.png', dpi=300)
            plt.close()

        # 2. Box Plot: Distribution Across All Stations
        corrections = df_agg['correction'].unique()

        for corr in corrections:
            df_corr = df_agg[df_agg['correction'] == corr].copy()
            df_corr['method_corr'] = df_corr['method']  # No need to add correction to label

            plt.figure(figsize=(14, 6))

            # RMSE subplot
            plt.subplot(1, 2, 1)
            box_data_rmse = [group['RMSE_allDOY'].values for _, group in df_corr.groupby('method_corr')]
            labels_rmse = df_corr['method_corr'].unique()
            plt.boxplot(box_data_rmse, labels=labels_rmse, showfliers=False)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'RMSE Distribution (Correction: {corr})')
            plt.ylabel('RMSE')

            # MAE subplot
            plt.subplot(1, 2, 2)
            box_data_mae = [group['MAE_allDOY'].values for _, group in df_corr.groupby('method_corr')]
            labels_mae = df_corr['method_corr'].unique()
            plt.boxplot(box_data_mae, labels=labels_mae, showfliers=False)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'MAE Distribution (Correction: {corr})')
            plt.ylabel('MAE')

            plt.suptitle(f'Distribution Across All Stations — Correction: {corr}')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'evaluation/visualizations/boxplot_all_stations_{corr}.png', dpi=300)
            plt.close()

        # 3. Heatmap: RMSE per Station vs Method
        corrections = df_agg['correction'].unique()

        for corr in corrections:
            df_corr = df_agg[df_agg['correction'] == corr]
            pivot_rmse = df_corr.pivot(index='station', columns='method', values='RMSE_allDOY').fillna(np.nan)

            plt.figure(figsize=(10, 8))
            im = plt.imshow(pivot_rmse.values, aspect='auto', interpolation='nearest', cmap='coolwarm')
            plt.colorbar(im, label='RMSE')
            plt.xticks(np.arange(len(pivot_rmse.columns)), pivot_rmse.columns, rotation=45, ha='right')
            plt.yticks(np.arange(len(pivot_rmse.index)), pivot_rmse.index)
            plt.title('Heatmap of Raw RMSE: Station vs Method')
            plt.tight_layout()
            plt.savefig(f'evaluation/visualizations/heatmap_rmse_{corr}.png', dpi=300)
            plt.close()

        # 4. Facet Grid / Small Multiples: One Bar Plot per Station
        stations_to_plot = ['kokee', 'hobart12', 'ishioka', 'onsala60', 'wark12m',
                              'wettzell', 'matera', 'fortleza', 'sejong', 'kokee12m', 'onsa13ne', 'onsa13sw']
        n_plots = len(stations_to_plot)
        n_cols = 3
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        for idx, st in enumerate(stations_to_plot):
            ax = axes[idx]
            df_st = df_agg[df_agg['station'] == st]
            methods = df_st['method'].unique()
            x = np.arange(len(methods))
            width = 0.25
            corrections = ['raw', 'glob-bias-corr', 'local-bias-corr']

            for i, corr in enumerate(corrections):
                df_corr = df_st[df_st['correction'] == corr]
                rmse_vals = [
                    df_corr[df_corr['method'] == m]['RMSE_allDOY'].values[0]
                    if not df_corr[df_corr['method'] == m].empty
                    else np.nan
                    for m in methods
                ]
                ax.bar(x + i * width - width, rmse_vals, width, label=corr, alpha=0.7)

            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.set_title(st.capitalize())
            ax.set_xlabel('Method')
            ax.set_ylabel('RMSE')
            ax.legend(fontsize='small')

        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig('evaluation/visualizations/facet_barplots.png', dpi=300)
        plt.close()

    # ---------- Medians (raw, global bias, local bias) ----------
    median_values = df.groupby(group_columns).median(numeric_only=True)
    median_values = median_values.reindex(group_keys)

    global_metrics = median_values[[
        'global_RMSE', 'global_MAE',
        'global_RMSE_globalbias', 'global_MAE_globalbias'
    ]].copy()

    station_cols = []
    for st in station_names:
        for suffix in ['', '_bcg', '_loc']:
            station_cols.append(f"{st}_RMSE{suffix}")
            station_cols.append(f"{st}_MAE{suffix}")
    station_cols = [c for c in station_cols if c in median_values.columns]
    station_metrics = median_values[station_cols].copy()

    global_metrics.index  = [name_map[k] for k in global_metrics.index]
    station_metrics.index = [name_map[k] for k in station_metrics.index]

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
