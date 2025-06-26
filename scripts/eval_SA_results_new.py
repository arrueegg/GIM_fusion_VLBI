import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from io import StringIO

warnings.filterwarnings("ignore")

# --- Constants & Settings ---
# Define the expected methods and pretty names
METHOD_MAP = {
    'GNSS': 'GNSS',
    'Fusion_1000_1': 'Fusion SW',
    'Fusion_1_1000': 'Fusion LW',
    'DTEC_100_1': 'DTEC SW',
    'DTEC_1_100': 'DTEC LW'
}
COMBINATIONS = []
for key in METHOD_MAP.keys():
    parts = key.split('_')
    if len(parts) == 3:
        approach, sw, lw = parts
        COMBINATIONS.append((approach, int(sw), int(lw)))
    else:
        COMBINATIONS.append((key, 1, 1))
MIN_STATION_COUNT = 50

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')

# --- VLBI meta data parsing ---

def parse_markdown_table(text: str, section_header: str) -> pd.DataFrame:
    pattern = re.escape(section_header) + r"(.*?)(\n##|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return pd.DataFrame()
    table_text = match.group(1).strip()
    # strip table separators and read as CSV
    lines = [l for l in table_text.splitlines()
             if not re.match(r"^\s*\|[-:\s|]+\|$", l)]
    cleaned = "\n".join([l.strip().strip("|") for l in lines if l.strip()])
    df = pd.read_csv(StringIO(cleaned), sep=r"\s*\|\s*", engine="python")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df

def load_vlbi_meta(vlbi_root: str, year: int, doy: int) -> pd.DataFrame:
    """
    Scan all VLBI sessions on that day, parse their summary.md → VTEC Time Series tables,
    and return a DataFrame with columns ['station','first_datetime','last_datetime'].
    """
    # build list of all SX/VGOS summary.md paths for doy and doy-1
    def doy_paths(subdir, year, doy):
        m = (pd.Timestamp(year,1,1) + pd.Timedelta(days=doy-1)).strftime("%Y%m%d")
        folder = os.path.join(vlbi_root, subdir, str(year))
        return [os.path.join(folder, p, "summary.md")
                for p in os.listdir(folder) if p.startswith(m+"-")]

    md_paths = set(doy_paths("SX", year, doy) +
                   doy_paths("VGOS", year, doy) +
                   doy_paths("SX", year, doy-1) +
                   doy_paths("VGOS", year, doy-1))

    all_dfs = []
    for fp in md_paths:
        if not os.path.exists(fp): continue
        raw = open(fp).read()
        vtec = parse_markdown_table(raw, "## VTEC Time Series")
        if not vtec.empty:
            vtec['datetime'] = pd.to_datetime(vtec['date'] + ' ' + vtec['epoch'])
            all_dfs.append(vtec)

    if not all_dfs:
        return pd.DataFrame(columns=['station','first_datetime','last_datetime'])

    combined = pd.concat(all_dfs, ignore_index=True)
    # min/max per station
    return (combined
            .groupby('station')['datetime']
            .agg(['min','max'])
            .reset_index()
            .rename(columns={'min':'first_datetime','max':'last_datetime'}))


# --- Data I/O & Parsing ---
def read_SA_metrics(folder, include_raw=False):
    """
    Compute daily metrics and optionally return raw data.

    Parameters
    ----------
    folder : str
        Path to SA_plots folder containing 'results.csv' and station CSVs.
    include_raw : bool
        If True, returns raw DataFrames for global and stations.

    Returns
    -------
    metrics : dict
        Daily global and per-station statistics.
    station_list : list
        Sorted station names.
    df_global : pandas.DataFrame, optional
        Raw global observations (only if include_raw=True).
    df_stations : dict of pandas.DataFrame, optional
        Raw station observations per station.
    """
    # ── Load VLBI meta data ───────────────────────────
    # Extract year and day of year from folder name
    match = re.search(r'_(\d{4})_(\d{3})_', folder)
    year = int(match.group(1))
    doy = int(match.group(2))
    # Load VLBI meta data for the given day
    vlbi_meta = load_vlbi_meta('/scratch2/arrueegg/WP1/VLBIono/Results/', year, doy)

    # ─── Global data ───

    # ── load global CSV ─────────────────────────────
    global_path = os.path.join(folder, 'results.csv')
    df_global = pd.read_csv(global_path, parse_dates=['time'])

    # ── global time‐filter ──────────────────────────
    if vlbi_meta.empty:
        # no overlap → empty for metrics
        df_global = pd.DataFrame(columns=df_global.columns)
        global_start = global_end = None
    else:
        global_start = vlbi_meta['first_datetime'].min()
        global_end   = vlbi_meta['last_datetime'].max()
        mask = (df_global['time'] >= global_start) & (df_global['time'] <= global_end)
        df_global = df_global.loc[mask].copy()

    # Raw residuals
    df_global['residuals_raw'] = df_global['model_prediction'] - df_global['vtec']

    # Estimate global bias (mean raw residual)
    b_global = df_global['residuals_raw'].mean()
    df_global['pred_bc_global']      = df_global['model_prediction'] - b_global
    df_global['residuals_bc_global'] = df_global['pred_bc_global'] - df_global['vtec']

    # Global metrics
    metrics = {
        # raw
        'global_bias_raw':  b_global,
        'global_rmse_raw':  np.sqrt((df_global['residuals_raw']**2).mean()),
        'global_mae_raw':   df_global['residuals_raw'].abs().mean(),
        'global_std_raw':   df_global['residuals_raw'].std(),
        'global_corr_raw':  df_global['model_prediction'].corr(df_global['vtec']),

        # bias-corrected (global)
        'global_rmse_bc_global':  np.sqrt((df_global['residuals_bc_global']**2).mean()),
        'global_mae_bc_global':   df_global['residuals_bc_global'].abs().mean(),
        'global_std_bc_global':   df_global['residuals_bc_global'].std(),
        'global_corr_bc_global':  df_global['pred_bc_global'].corr(df_global['vtec']),
    }

    # ─── Station data ───
    station_data = {}
    df_stations = {}
    for path in glob.glob(os.path.join(folder, 'station_*_results.csv')):
        st_match = re.search(r'station_(.+?)_results\.csv$', path)
        if not st_match:
            continue
        st = st_match.group(1).lower()
        df_st = pd.read_csv(path, parse_dates=['time'])
        if 'residuals_raw' not in df_st.columns:
            df_st['residuals_raw'] = df_st['model_prediction'] - df_st['vtec']
        df_stations[st] = df_st.copy()

        d = {'count': len(df_st)}
        # Raw
        raw = df_st['residuals_raw']
        d['rmse_raw'] = np.sqrt((raw**2).mean())
        d['mae_raw']  = raw.abs().mean()
        # Global bias-corrected
        if 'residuals_bc_global' in df_st:
            gb = df_st['residuals_bc_global']
            d['rmse_bc_global'] = np.sqrt((gb**2).mean())
            d['mae_bc_global']  = gb.abs().mean()
        # Local bias-corrected
        if 'residuals_bc_local' in df_st:
            lb = df_st['residuals_bc_local']
            d['rmse_bc_local'] = np.sqrt((lb**2).mean())
            d['mae_bc_local']  = lb.abs().mean()
        station_data[st] = d

    station_list = sorted(station_data.keys())
    metrics['station_data'] = station_data
    metrics['station_list'] = station_list

    if include_raw:
        return metrics, station_list, df_global, df_stations
    return metrics, station_list


def collect_metrics(experiments_folder):
    """
    Loop through experiment folders (methods), compute metrics and aggregate raw data across all days.

    Returns
    -------
    df_metrics : pandas.DataFrame
        Aggregated metrics per method.
    df_global_all : pandas.DataFrame
        Concatenated global raw observations across all methods.
    df_station_all : dict of pandas.DataFrame
        Concatenated station raw observations per station across methods.
    """
    metrics_list = []
    global_list = []
    station_all = {}

    #for method in sorted(os.listdir(experiments_folder), key=lambda x: int(re.search(r'_(\d{4}_\d{3})_', x).group(1).replace('_', '')) if re.search(r'_(\d{4}_\d{3})_', x) else float('inf'))[:10]:
    for method in sorted(os.listdir(experiments_folder)):
        sa_folder = os.path.join(experiments_folder, method, 'SA_plots')
        if not os.path.isdir(sa_folder):
            continue
        global_csv = os.path.join(sa_folder, 'results.csv')
        if not os.path.isfile(global_csv):
            continue
        # Read metrics and raw data
        metrics, stations, df_g, dfs = read_SA_metrics(sa_folder, include_raw=True)
        # Per-method metrics summary
        rec = {'method': method,
               'global_bias': metrics['global_bias_raw'],
               'global_RMSE': metrics['global_rmse_raw'],
               'global_MAE':  metrics['global_mae_raw']}
        for st in stations:
            st_data = metrics['station_data'][st]
            rec.update({
                f'{st}_N':             st_data['count'],
                f'{st}_RMSE':          st_data['rmse_raw'],
                f'{st}_MAE':           st_data['mae_raw'],
                f'{st}_RMSE_bc_global': st_data.get('rmse_bc_global', np.nan),
                f'{st}_MAE_bc_global':  st_data.get('mae_bc_global', np.nan),
                f'{st}_RMSE_bc_local':  st_data.get('rmse_bc_local', np.nan),
                f'{st}_MAE_bc_local':   st_data.get('mae_bc_local', np.nan)
            })
        metrics_list.append(rec)
        # Tag and collect raw global data
        df_g = df_g.copy()
        df_g['method'] = method
        global_list.append(df_g)
        # Tag and collect raw station data
        for st, df_st in dfs.items():
            dfc = df_st.copy()
            dfc['method'] = method
            station_all.setdefault(st, []).append(dfc)

    df_metrics = pd.DataFrame(metrics_list)
    df_global_all = pd.concat(global_list, ignore_index=True) if global_list else pd.DataFrame()
    df_station_all = {st: pd.concat(lst, ignore_index=True) for st, lst in station_all.items()}
    df_metrics.to_csv('evaluation/SA_metrics_summary.csv', index=False)
    return df_metrics, df_global_all, df_station_all

# --- Plotting & Evaluation Functions ---

def plot_global_annual_box(df_global_all, out_dir='evaluation/global_annual_boxplots'):
    os.makedirs(out_dir, exist_ok=True)
    corrections = [
        ('residuals_raw',       'Raw'),
        ('residuals_bc_global', 'Global Bias Corrected'),
        ('residuals_bc_local',  'Local Bias Corrected')
    ]

    median_values = {}  # Dictionary to store median values

    for col, label in corrections:
        if col not in df_global_all.columns:
            continue

        # Extract approach, SW number, and LW number from the 'method' column
        df_global_all['approach'] = df_global_all['method'].apply(
            lambda x: 'DTEC' if 'DTEC' in x else ('Fusion' if 'Fusion' in x else ('GNSS' if 'GNSS' in x else 'Unknown'))
        )
        df_global_all['SW'] = df_global_all['method'].str.extract(r'_SW(\d+)', expand=False).astype(int)
        df_global_all['LW'] = df_global_all['method'].str.extract(r'_LW(\d+)', expand=False).astype(int)    

        data = [
            df_global_all[
                (df_global_all['approach']==approach) &
                (df_global_all['SW']==sw) &
                (df_global_all['LW']==lw)
            ][col].dropna()
            for approach, sw, lw in COMBINATIONS
        ]

        fig, ax = plt.subplots(figsize=(12,8), dpi=300)

        # default line widths & colors for medians, whiskers, caps
        bp = ax.boxplot(
            data,
            widths=0.6,
            labels=METHOD_MAP.values(),
            notch=False,
            patch_artist=True,
            showfliers=False,
            boxprops     = dict(facecolor='tab:blue', edgecolor='black', linewidth=1),
            whiskerprops = dict(color='black', linewidth=2),
            capprops     = dict(color='black', linewidth=2),
            medianprops  = dict(color='tab:orange', linewidth=3)
        )

        # now draw your full-width orange median
        medians = []
        for box, median in zip(bp['boxes'], bp['medians']):
            verts = box.get_path().vertices
            x0, x1 = verts[:,0].min(), verts[:,0].max()
            y       = median.get_ydata()[0]
            median.set_xdata([x0, x1])
            median.set_ydata([y,  y])
            median.set(color='tab:orange', linewidth=3)
            medians.append(y)  # Store the median value

        # Store median values for the current correction
        median_values[label] = medians

        #ax.set_xlabel('Method', fontsize=26)
        ax.set_ylabel('Residual [TECU]', fontsize=26)
        plt.xticks(rotation=0, ha='center', fontsize=26)
        plt.yticks(fontsize=26)

        plt.tight_layout()
        fname = f'global_{label.lower().replace(" ","_")}_by_method_notitle.png'
        plt.savefig(os.path.join(out_dir, fname))
        ax.set_title(f'Global {label} Residuals', fontsize=30, weight='bold')
        plt.tight_layout()
        fname = f'global_{label.lower().replace(" ","_")}_by_method.png'
        plt.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.close()

    # Save median values to a CSV file
    median_df = pd.DataFrame(median_values, index=METHOD_MAP.values())
    median_df.to_csv(os.path.join(out_dir, 'global_median_values.csv'))


def plot_station_annual_box(df_station_all, out_base='evaluation/annual_station_boxplots'):
    corrections = [
        ('residuals_raw',       'Raw'),
        ('residuals_bc_global', 'Global Bias Corrected'),
        ('residuals_bc_local',  'Local Bias Corrected')
    ]

    # treat all stations as one combined if requested
    df_station_all['all_stations'] = pd.concat(df_station_all.values(), ignore_index=True)

    for st, df in df_station_all.items():
        out_dir = os.path.join(out_base, st)
        os.makedirs(out_dir, exist_ok=True)

        for col, label in corrections:
            if col not in df.columns:
                continue

            # extract approach/SW/LW
            df['approach'] = df['method'].apply(
                lambda x: 'DTEC'   if 'DTEC' in x
                             else 'Fusion' if 'Fusion' in x
                             else 'GNSS'   if 'GNSS' in x
                             else 'Unknown'
            )
            df['SW'] = df['method'].str.extract(r'_SW(\d+)', expand=False).astype(int)
            df['LW'] = df['method'].str.extract(r'_LW(\d+)', expand=False).astype(int)

            # build one series per method
            data = [
                df[
                    (df['approach']==approach) &
                    (df['SW']==sw) &
                    (df['LW']==lw)
                ][col].dropna()
                for approach, sw, lw in COMBINATIONS
            ]

            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

            # draw the boxes
            bp = ax.boxplot(
                data,
                widths=0.6,
                labels=METHOD_MAP.values(),
                notch=False,
                patch_artist=True,
                showfliers=False,
                boxprops     = dict(facecolor='tab:blue', edgecolor='black', linewidth=1),
                whiskerprops = dict(color='black', linewidth=2),
                capprops     = dict(color='black', linewidth=2),
                medianprops  = dict(color='tab:orange', linewidth=3)
            )

            # stretch each median to the box edges
            for box, median in zip(bp['boxes'], bp['medians']):
                verts = box.get_path().vertices
                x0, x1 = verts[:,0].min(), verts[:,0].max()
                y       = median.get_ydata()[0]
                median.set_xdata([x0, x1])
                median.set_ydata([y,  y])
                median.set(color='tab:orange', linewidth=3)

            # labels & titles
            station_name = 'All Stations' if st=='all_stations' else st.capitalize()
            #ax.set_xlabel('Method', fontsize=26)
            ax.set_ylabel('Residual [TECU]', fontsize=26)
            ax.tick_params(axis='x', labelsize=26)
            ax.tick_params(axis='y', labelsize=26)

            plt.tight_layout()
            fname = f'{st}_{label.lower().replace(" ","_")}_by_method_notitle.png'
            plt.savefig(os.path.join(out_dir, fname))
            ax.set_title(f'{station_name} {label} Residuals', fontsize=30, weight='bold')
            plt.tight_layout()
            fname = f'{st}_{label.lower().replace(" ","_")}_by_method.png'
            plt.savefig(os.path.join(out_dir, fname), dpi=300)
            plt.close()


def compute_annual_station_metrics(df_station_all):
    records = []
    for st, df in df_station_all.items():
        for col, corr in [('residuals_raw', 'raw'),
                          ('residuals_bc_global', 'global_bc'),
                          ('residuals_bc_local', 'local_bc')]:
            if col in df.columns:
                arr = df[col].dropna().to_numpy()
                for approach, sw, lw in COMBINATIONS:
                    arr_combination = df[
                        (df['approach'] == approach) &
                        (df['SW'] == sw) &
                        (df['LW'] == lw)
                    ][col].dropna().to_numpy()
                    records.append({'station': st, 'correction': corr, 'approach': approach,
                                    'SW': sw, 'LW': lw,
                                    'RMSE': np.sqrt((arr_combination**2).mean()),
                                    'MAE':  np.mean(np.abs(arr_combination)), 
                                    'mean': np.mean(arr_combination),
                                    'median': np.median(arr_combination),
                                    'count': len(arr_combination)})
    df_yearly_metrics = pd.DataFrame(records)
    df_yearly_metrics.to_csv('evaluation/annual_station_metrics.csv', index=False)
    return df_yearly_metrics


def plot_annual_barplots(df_yearly_metrics, out_dir='evaluation/annual_visualizations'):
    os.makedirs(out_dir, exist_ok=True)
    stations = df_yearly_metrics['station'].unique()
    corrs = df_yearly_metrics['correction'].unique()
    x = np.arange(len(stations))
    width = 0.25
    # RMSE
    plt.figure(figsize=(12,6))
    for i, corr in enumerate(corrs):
        vals = df_yearly_metrics[df_yearly_metrics['correction']==corr]
        vals = vals.drop_duplicates(subset='station').set_index('station').reindex(stations)['RMSE']
        plt.bar(x + (i-1)*width, vals, width, label=corr)
    plt.xticks(x, stations, rotation=45, ha='right')
    plt.ylabel('RMSE')
    plt.title('Annual RMSE by Station & Correction')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'annual_rmse_barplot.png'), dpi=300)
    plt.close()
    # MAE
    plt.figure(figsize=(12,6))
    for i, corr in enumerate(corrs):
        vals = df_yearly_metrics[df_yearly_metrics['correction']==corr]
        vals = vals.drop_duplicates(subset='station').set_index('station').reindex(stations)['MAE']
        plt.bar(x + (i-1)*width, vals, width, label=corr)
    plt.xticks(x, stations, rotation=45, ha='right')
    plt.ylabel('MAE')
    plt.title('Annual MAE by Station & Correction')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'annual_mae_barplot.png'), dpi=300)
    plt.close()


def plot_annual_heatmap(df_yearly_metrics, out_dir='evaluation/annual_visualizations'):
    os.makedirs(out_dir, exist_ok=True)

    for corr, group in df_yearly_metrics.groupby('correction'):
        # 1) Reconstruct the exact method key from approach, SW, LW:
        group = group.copy()
        group['method_key'] = group.apply(
            lambda row: f"{row['approach']}_{row['SW']}_{row['LW']}" 
            if row['approach'] != 'GNSS' else 'GNSS', axis=1
        )

        # 2) Pivot on station × method_key, taking RMSE
        pivot = group.pivot(
            index='station',
            columns='method_key',
            values='RMSE'
        )
        # 3) Reorder to your METHOD_MAP order, and rename to the pretty labels
        pivot = pivot.reindex(columns=METHOD_MAP.keys())
        pivot.columns = [METHOD_MAP[m] for m in pivot.columns]

        # 4) Plot
        plt.figure(figsize=(8, 8))
        im = plt.imshow(pivot, aspect='auto', cmap='coolwarm')
        plt.colorbar(im, label='RMSE')
        plt.xticks(np.arange(len(pivot.columns)),
                   pivot.columns,
                   rotation=45,
                   ha='right')
        plt.yticks(np.arange(len(pivot.index)),
                   pivot.index)
        plt.title(f'Heatmap of Annual RMSE by Station & Approach\n(Correction: {corr})')
        plt.tight_layout()
        fname = f'annual_rmse_heatmap_{corr}.png'
        plt.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.close()



def evaluate(df_metrics, df_global_all, df_station_all):
    """Run full evaluation: annual global and station boxplots, plus aggregated visuals."""
    plot_global_annual_box(df_global_all)
    plot_station_annual_box(df_station_all)
    df_yearly_metrics = compute_annual_station_metrics(df_station_all)
    plot_annual_barplots(df_yearly_metrics)
    plot_annual_heatmap(df_yearly_metrics)


def main():
    experiments_folder = '/scratch2/arrueegg/WP2/GIM_fusion_VLBI/experiments/'
    df_metrics, df_global_all, df_station_all = collect_metrics(experiments_folder)
    print(f"Loaded metrics for {int(len(df_metrics)/5)} days")
    evaluate(df_metrics, df_global_all, df_station_all)
    print("Evaluation completed and saved to 'evaluation/' directory.")

if __name__ == "__main__":
    main()
