#!/usr/bin/env python3
import os
import time
import argparse
import yaml
import logging
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import re
from io import StringIO
import warnings

from utils.locationencoder.pe import SphericalHarmonics
from utils.config_parser import parse_config
from models.model import get_model

warnings.filterwarnings("ignore")

# Station definitions for per-station metrics
# ----------------------------------------------------------------------------
# Mapping from VLBI station codes (Station_Name) → (Latitude, Longitude)
# as extracted from ivstrf_stations_with_location_names.csv, 
# using the coordinates you supplied in STATION_COORDS.
# ----------------------------------------------------------------------------

STATION_COORDS = {
    # Kokee (22.13, -159.66)  
    'KAUAI':    (22.13,   -159.66),
    'KOKEE':    (22.13,   -159.66),
    'KOKEE12M': (22.13,   -159.66),

    # Hobart (−42.80, 147.44)
    'HOBART26': ( -42.80,  147.44),
    'HOBART12': ( -42.80,  147.44),

    # Ishioka (36.21, 140.22)
    'ISHIOKA':  (36.21,   140.22),

    # Santa Maria (36.98, −25.16)
    'RAEGSMAR': (36.98,   -25.16),

    # Onsala (57.40, 11.92)
    'ONSALA60':  (57.40,  11.92),
    'ONSA13NE':  (57.40,  11.92),
    'ONSA13SW':  (57.40,  11.92),

    # Wettzell (49.15, 12.88)
    'WETTZELL':  (49.15,  12.88),
    'TIGOWTZL':  (49.15,  12.88),
    'WETTZ13N':  (49.15,  12.88),
    'WETTZ13S':  (49.15,  12.88),

    # Yebes (40.52, −3.09)
    'YEBES':     (40.52,  -3.09),
    'YEBES40M':  (40.52,  -3.09),
    'RAEGYEB':   (40.52,  -3.09),

    # Matera (40.65, 16.70)
    'MATERA':    (40.65,  16.70),

    # Sejong (36.52, 127.30)
    'SEJONG':    (36.52, 127.30),

    # Warkworth (−36.43, 174.66)
    'WARK12M':   (-36.43, 174.66),

    # Westford (42.61, −71.49)
    'HAYSTACK':  (42.61,  -71.49),
    'WESTFORD':  (42.61,  -71.49),

    # Forteleza (−3.88, −38.43)
    'FORTLEZA':  (-3.88,  -38.43),
}

RADIUS_KM = 1000.0  # buffer around each station
LOSS_FN = "LaplaceLoss"

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance (km) between two points.
    """
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def parse_markdown_table(text: str, section_header: str) -> pd.DataFrame:
    """
    Given a section header (e.g. "## VTEC Time Series"), extracts the markdown table
    that follows (until the next header "##" or end-of-file) and returns it as a DataFrame.
    """
    pattern = re.escape(section_header) + r"(.*?)(\n##|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return pd.DataFrame()
    table_text = match.group(1).strip()
    if not table_text:
        return pd.DataFrame()
    # Split into lines and remove any lines that are table separators
    lines = table_text.splitlines()
    data_lines = [line for line in lines if not re.match(r"^\s*\|[-:\s|]+\|$", line)]
    cleaned = "\n".join([line.strip().strip("|") for line in data_lines if line.strip()])
    if not cleaned:
        return pd.DataFrame()
    try:
        df = pd.read_csv(StringIO(cleaned), sep=r"\s*\|\s*", engine="python")
    except Exception as e:
        print(f"Error parsing table for section '{section_header}': {e}")
        return pd.DataFrame()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df

def load_vlbi_meta(cfg):
    """
    Load VLBI metadata for a given year and day of year.
    """
    year, doy1, doy2 = cfg['year'], cfg['doy'], cfg['doy'] - 1
    vlbi_path = cfg['data']['VLBI_data_path']
    month1, day1 = (pd.Timestamp(year, 1, 1) + pd.Timedelta(days=doy1 - 1)).strftime('%m %d').split()
    month2, day2 = (pd.Timestamp(year, 1, 1) + pd.Timedelta(days=doy2 - 1)).strftime('%m %d').split()
    paths_sx1 = [os.path.join(vlbi_path, 'SX', str(year), p) for p in os.listdir(os.path.join(vlbi_path, 'SX', str(year))) 
                if p.startswith(f"{year}{month1}{day1}-")]
    paths_sx2 = [os.path.join(vlbi_path, 'SX', str(year), p) for p in os.listdir(os.path.join(vlbi_path, 'SX', str(year)))
                if p.startswith(f"{year}{month2}{day2}-")]
    paths_vgos1 = [os.path.join(vlbi_path, 'VGOS', str(year), p) for p in os.listdir(os.path.join(vlbi_path, 'VGOS', str(year))) 
                  if p.startswith(f"{year}{month1}{day1}-")]
    paths_vgos2 = [os.path.join(vlbi_path, 'VGOS', str(year), p) for p in os.listdir(os.path.join(vlbi_path, 'VGOS', str(year)))
                  if p.startswith(f"{year}{month2}{day2}-")]
    paths = set(paths_sx1 + paths_vgos1 + paths_sx2 + paths_vgos2)

    all_data = []
    for p in paths:
        fp = os.path.join(p, 'summary.md')
        if not os.path.exists(fp):
            continue
        with open(fp, "r") as f:
            content = f.read()
        vtec_df = parse_markdown_table(content, "## VTEC Time Series")
        all_data.append(vtec_df)
    
    all_data = pd.concat(all_data, ignore_index=True)
    # Convert 'epoch' to a datetime column assuming 'date' contains the date information
    all_data['datetime'] = pd.to_datetime(all_data['date'] + ' ' + all_data['epoch'])

    # Extract unique station names and their corresponding first and last datetime
    station_epochs = all_data.groupby('station')['datetime'].agg(['min', 'max']).reset_index()
    station_epochs.rename(columns={'min': 'first_datetime', 'max': 'last_datetime'}, inplace=True)
    
    return station_epochs

def load_data(csv_file, doy):
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    df = df[df['time'].dt.dayofyear == doy]
    return df

def prepare_inputs(sa_data, device, sh_encoder):
    lat = torch.tensor(sa_data['sm_lat'].values, dtype=torch.float32, device=device)
    lon = torch.tensor(sa_data['sm_lon'].values, dtype=torch.float32, device=device)
    inputs = torch.stack([lon, lat], dim=1)
    if sh_encoder:
        inputs = sh_encoder(inputs)
    times = (sa_data['time'].dt.hour * 3600 +
             sa_data['time'].dt.minute * 60 +
             sa_data['time'].dt.second)
    sod = torch.tensor(times.values, dtype=torch.float32, device=device)
    sin_utc = torch.sin(sod / 86400 * 2 * torch.pi)
    cos_utc = torch.cos(sod / 86400 * 2 * torch.pi)
    norm = 2 * sod / 86400 - 1
    return torch.cat([inputs,
                      sin_utc.unsqueeze(1),
                      cos_utc.unsqueeze(1),
                      norm.unsqueeze(1)],
                     dim=1)


def calculate_metrics(config, results, vlbi_meta):
    """
    Compute global and per-station metrics, applying:
    1) A global time filter (union of all VLBI windows) before global metrics.
    2) Per-station spatio-temporal filter for station metrics.
    Writes all metrics into 'metrics.txt' under SA_plots/.
    Returns a dictionary of keys needed by plot_results().
    """

    # ----------------------------------------------------------------------
    # 1) GLOBAL TIME FILTER: find union of all VLBI windows
    if vlbi_meta.empty:
        # No VLBI data available: global metrics must be "N/A"
        global_start, global_end = None, None
        results_global = pd.DataFrame(columns=results.columns)
    else:
        global_start = vlbi_meta['first_datetime'].min()
        global_end = vlbi_meta['last_datetime'].max()
        # Filter altimetry rows to only times within [global_start, global_end]
        mask_global = (results['time'] >= global_start) & (results['time'] <= global_end)
        results_global = results.loc[mask_global].copy()
        plot_altimetry_map(config, results_global)

    # ----------------------------------------------------------------------
    # 2) GLOBAL METRICS (RAW and GLOBAL-BIAS-CORRECTED) on results_global

    if results_global.empty:
        # If no overlapping altimetry, mark all global metrics as None
        rmse_raw = mae_raw = std_raw = corr_raw = None
        bias_global = None
        rmse_bcg = mae_bcg = std_bcg = corr_bcg = None
    else:
        # 2a) Raw residuals for global set
        results_global['residuals_raw'] = results_global['model_prediction'] - results_global['vtec']
        rmse_raw = round(np.sqrt(np.mean(results_global['residuals_raw']**2)), 2)
        mae_raw = round(np.mean(np.abs(results_global['residuals_raw'])), 2)
        std_raw = round(np.std(results_global['residuals_raw']), 2)
        corr_raw = round(np.corrcoef(results_global['model_prediction'], results_global['vtec'])[0, 1], 3)

        # 2b) Compute global bias, then bias-corrected residuals
        bias_global = np.mean(results_global['residuals_raw'])
        results_global['residuals_bc_global'] = results_global['residuals_raw'] - bias_global
        rmse_bcg = round(np.sqrt(np.mean(results_global['residuals_bc_global']**2)), 2)
        mae_bcg = round(np.mean(np.abs(results_global['residuals_bc_global'])), 2)
        std_bcg = round(np.std(results_global['residuals_bc_global']), 2)
        corr_bcg = round(np.corrcoef(
            results_global['model_prediction'] - bias_global,
            results_global['vtec']
        )[0, 1], 3)

    # ----------------------------------------------------------------------
    # 3) PER-STATION METRICS
    station_stats = []
    for station, (slat, slon) in STATION_COORDS.items():
        # 3a) Get this station's own VLBI window
        row = vlbi_meta[vlbi_meta['station'] == station]
        if row.empty:
            # No VLBI record for this station on that day
            station_stats.append({
                'station':      station,
                'count':        0,
                'RMSE_raw':     None,
                'MAE_raw':      None,
                'RMSE_globCor': None,
                'MAE_globCor':  None,
                'RMSE_locCor':  None,
                'MAE_locCor':   None
            })
            continue

        start_s = row['first_datetime'].iloc[0]
        end_s = row['last_datetime'].iloc[0]

        # 3b) Spatial mask: within RADIUS_KM of this station
        dists = haversine(
            slat, slon,
            results['lat'].values,
            results['lon'].values
        )
        mask_space = (dists <= RADIUS_KM)

        # 3c) Temporal mask: only times within [start_s, end_s]
        mask_time_s = (results['time'] >= start_s) & (results['time'] <= end_s)

        # 3d) Intersection: only altimetry points near station _and_ during its VLBI window
        subset = results.loc[mask_space & mask_time_s].copy()

        if subset.empty:
            # No overlapping altimetry points for this station
            station_stats.append({
                'station':      station,
                'count':        0,
                'RMSE_raw':     None,
                'MAE_raw':      None,
                'RMSE_globCor': None,
                'MAE_globCor':  None,
                'RMSE_locCor':  None,
                'MAE_locCor':   None
            })
            continue

        # 3e) Compute raw station metrics on subset
        subset['residuals_raw'] = subset['model_prediction'] - subset['vtec']
        rmse_raw_s = round(np.sqrt(np.mean(subset['residuals_raw']**2)), 2)
        mae_raw_s = round(np.mean(np.abs(subset['residuals_raw'])), 2)

        # 3f) Compute global-bias-corrected station metrics
        if bias_global is not None:
            subset['residuals_bc_global'] = subset['residuals_raw'] - bias_global
            rmse_gs = round(np.sqrt(np.mean(subset['residuals_bc_global']**2)), 2)
            mae_gs = round(np.mean(np.abs(subset['residuals_bc_global'])), 2)
        else:
            rmse_gs = mae_gs = None

        # 3g) Compute station's own local bias and local-bias-corrected metrics
        local_bias = np.mean(subset['residuals_raw'])
        subset['residuals_bc_local'] = subset['residuals_raw'] - local_bias
        rmse_ls = round(np.sqrt(np.mean(subset['residuals_bc_local']**2)), 2)
        mae_ls = round(np.mean(np.abs(subset['residuals_bc_local'])), 2)

        station_stats.append({
            'station':      station,
            'count':        len(subset),
            'RMSE_raw':     rmse_raw_s,
            'MAE_raw':      mae_raw_s,
            'RMSE_globCor': rmse_gs,
            'MAE_globCor':  mae_gs,
            'RMSE_locCor':  rmse_ls,
            'MAE_locCor':   mae_ls
        })

        subset.to_csv(
            os.path.join(config['output_dir'], 'SA_plots', f'station_{station}_results.csv'),
            index=False
        )

    # ----------------------------------------------------------------------
    # 4) WRITE EVERYTHING INTO metrics.txt
    out_dir = os.path.join(config['output_dir'], 'SA_plots')
    os.makedirs(out_dir, exist_ok=True)
    metrics_file = os.path.join(out_dir, 'metrics.txt')

    with open(metrics_file, 'w') as f:
        # Global bias (if any)
        if bias_global is None:
            f.write("GLOBAL BIAS: N/A (no overlapping VLBI vs altimetry)\n\n")
        else:
            f.write(f"GLOBAL BIAS: {round(bias_global, 2)}\n\n")

        # Raw global metrics
        f.write("RAW GLOBAL METRICS:\n")
        if rmse_raw is None:
            f.write("  RMSE: N/A\n  MAE: N/A\n  STD: N/A\n  Correlation: N/A\n\n")
        else:
            f.write(f"  RMSE:        {rmse_raw}\n")
            f.write(f"  MAE:         {mae_raw}\n")
            f.write(f"  STD:         {std_raw}\n")
            f.write(f"  Correlation: {corr_raw}\n\n")

        # Global-bias-corrected global metrics
        f.write("GLOBAL-BIAS-CORRECTED GLOBAL METRICS:\n")
        if rmse_bcg is None:
            f.write("  RMSE: N/A\n  MAE: N/A\n  STD: N/A\n  Correlation: N/A\n\n")
        else:
            f.write(f"  RMSE:        {rmse_bcg}\n")
            f.write(f"  MAE:         {mae_bcg}\n")
            f.write(f"  STD:         {std_bcg}\n")
            f.write(f"  Correlation: {corr_bcg}\n\n")

        # Per-station header
        f.write(f"PER-STATION METRICS (within {RADIUS_KM:.0f} km of station, during station VLBI window):\n")
        f.write("  {:<12s} {:>6s} {:>10s} {:>10s} {:>12s} {:>12s} {:>12s} {:>12s}\n"
                .format("Station", "Count",
                        "RMSE_raw", "MAE_raw",
                        "RMSE_globCor", "MAE_globCor",
                        "RMSE_locCor",  "MAE_locCor"))

        # Per-station rows
        for s in station_stats:
            f.write("  {:<12s} {:>6d} {:>10s} {:>10s} {:>12s} {:>12s} {:>12s} {:>12s}\n".format(
                s['station'],
                s['count'],
                str(s['RMSE_raw']) if s['RMSE_raw'] is not None else "   N/A",
                str(s['MAE_raw'])  if s['MAE_raw']  is not None else "   N/A",
                str(s['RMSE_globCor']) if s['RMSE_globCor'] is not None else "   N/A",
                str(s['MAE_globCor'])  if s['MAE_globCor']  is not None else "   N/A",
                str(s['RMSE_locCor']) if s['RMSE_locCor'] is not None else "   N/A",
                str(s['MAE_locCor'])  if s['MAE_locCor']  is not None else "   N/A"
            ))

    # ----------------------------------------------------------------------
    # 5) Return keys for plotting or further use
    return {
        'BIAS':      (None if bias_global is None else round(bias_global, 2)),
        'RMSE_raw':  rmse_raw,  'MAE_raw':  mae_raw,  'STD_raw':  std_raw,  'CORR_raw':  corr_raw,
        'RMSE_bcg':  rmse_bcg,  'MAE_bcg':  mae_bcg,  'STD_bcg':  std_bcg,  'CORR_bcg':  corr_bcg
    }

def plot_altimetry_map(config, results):
    # -----------------------------------------------------------------------------
    # 1. Set up the figure and a Cartopy GeoAxes (PlateCarree equirectangular)
    # -----------------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 5))
    ax  = plt.axes(projection=ccrs.PlateCarree())

    # -----------------------------------------------------------------------------
    # 2. Draw the basemap (coastlines + optional land/ocean shading)
    # -----------------------------------------------------------------------------
    ax.stock_img()  # simple shaded relief + coastlines
    ax.coastlines(resolution='110m', linewidth=0.5)

    # -----------------------------------------------------------------------------
    # 3. Draw gridlines (optional)
    # -----------------------------------------------------------------------------
    gl = ax.gridlines(draw_labels=True,
                    linewidth=0.3,
                    color='gray',
                    alpha=0.5,
                    linestyle='--')
    gl.top_labels    = False
    gl.right_labels  = False

    # -----------------------------------------------------------------------------
    # 4. Overlay your scatter plot
    # -----------------------------------------------------------------------------
    ax.scatter(
        results.loc[:,'lon'],
        results.loc[:,'lat'],
        s=16,                   # marker size
        c='red',                # marker color
        alpha=0.7,              # transparency
        transform=ccrs.PlateCarree(),
    )

    # -----------------------------------------------------------------------------
    # 5. Add title, legend, etc.
    # -----------------------------------------------------------------------------
    year = results['time'].dt.year.iloc[0]
    doy = results['time'].dt.dayofyear.iloc[0]
    plt.title(f"Jason-3 observation during VLBI session on {year}-{doy:03d}",)

    plt.savefig(os.path.join(config['output_dir'], 'SA_plots', 'altimetry_map.png'),
                bbox_inches='tight', dpi=300)

def plot_results(config, results, metrics, vlbi_meta):
    """
    Plot:
      1) Predictions vs Ground Truth (raw)
      2) Raw residual histogram
      3) Global‐bias‐corrected residual histogram
    Uses exactly the columns produced by calculate_metrics().
    """
    out_dir = os.path.join(config['output_dir'], 'SA_plots')

    # 1) Predictions vs Ground Truth (raw)
    plt.figure(figsize=(8, 8))
    plt.scatter(results['model_prediction'], results['vtec'], s=0.1, alpha=0.3)
    mn, mx = results['vtec'].min(), results['vtec'].max()
    plt.plot([mn, mx], [mn, mx], 'r--')
    text_raw = "N/A" if metrics['RMSE_raw'] is None else f"RMSE: {metrics['RMSE_raw']}\nMAE: {metrics['MAE_raw']}"
    plt.text(
        0.05, 0.95,
        text_raw,
        transform=plt.gca().transAxes, va='top',
        bbox=dict(boxstyle='round', fc='white')
    )
    plt.xlabel('Model Prediction')
    plt.ylabel('VTEC (Ground Truth)')
    plt.savefig(os.path.join(out_dir, 'pred_vs_gt.png'))
    plt.close()

    # 2) Raw residual histogram
    plt.figure(figsize=(8, 8))
    residuals_raw = results['model_prediction'] - results['vtec']
    plt.hist(residuals_raw, bins=100, alpha=0.75, edgecolor='black', linewidth=0.5)
    text_raw_hist = "N/A" if metrics['RMSE_raw'] is None else f"RMSE: {metrics['RMSE_raw']}\nMAE: {metrics['MAE_raw']}"
    plt.text(
        0.05, 0.95,
        text_raw_hist,
        transform=plt.gca().transAxes, va='top',
        bbox=dict(boxstyle='round', fc='white')
    )
    plt.xlabel('Residuals (raw)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(out_dir, 'residuals_hist.png'))
    plt.close()


def evaluate_single(config, mode, sw, lw):
    # Update config for this run
    cfg = config.copy()
    cfg['mode'] = mode
    if sw is not None:
        cfg['vlbi_sampling_weight'] = sw
    if lw is not None:
        cfg['vlbi_loss_weight'] = lw
    cfg['loss_fn'] = LOSS_FN

    # Set output directory based on logic
    exp_root = cfg.get('output_dir')
    tag = f"SW{int(sw) if sw else 1}_LW{int(lw) if lw else 1}"
    out_dir = os.path.join(exp_root, f"{mode}_{cfg['year']}_{cfg['doy']:03d}_{tag}")
    cfg['output_dir'] = out_dir

    # Skip if metrics already exist (unless forced)
    metrics_path = os.path.join(out_dir, 'SA_plots', 'metrics.txt')
    if os.path.exists(metrics_path) and not cfg.get('force', False):
        logging.info(f"Skipping {mode} {tag}, metrics exists")
        return

    logging.info(f"Evaluating {mode} with {tag}")
    t0 = time.time()

    # 1) Load VLBI metadata for this day
    vlbi_meta = load_vlbi_meta(cfg)

    # 2) Load and filter altimetry/GNSS data
    csv_path = os.path.join(cfg['data']['GNSS_data_path'], f"sa_dataset.csv")
    if 'cluster' in csv_path:
        # Replace with cluster‐specific path if needed
        csv_path = '/cluster/work/igp_psr/arrueegg/sa_dataset.csv'
    sa_data = load_data(csv_path, cfg['doy'])

    # 3) Prepare model inputs
    sh_enc = None
    if cfg['preprocessing'].get('SH_encoding'):
        sh_enc = SphericalHarmonics(cfg['preprocessing']['SH_degree']).to(cfg['device'])
    inputs = prepare_inputs(sa_data, cfg['device'], sh_enc)

    # 4) Ensemble inference
    preds = []
    model_dir = os.path.join(out_dir, 'model')
    if os.path.isdir(model_dir):
        for fn in os.listdir(model_dir):
            m = get_model(cfg).to(cfg['device'])
            state = torch.load(os.path.join(model_dir, fn), map_location=cfg['device'])
            m.load_state_dict(state['model_state_dict'])
            m.eval()
            with torch.no_grad():
                preds.append(m(inputs).cpu().numpy())
    if not preds:
        logging.warning(f"No model files found in {model_dir}; skipping inference.")
        return
    ensemble = np.mean(preds, axis=0)

    # 5) Save results DataFrame
    res = sa_data.copy()
    res['model_prediction'] = ensemble[:, 0]
    os.makedirs(os.path.join(out_dir, 'SA_plots'), exist_ok=True)
    res.to_csv(os.path.join(out_dir, 'SA_plots', 'results.csv'), index=False)

    # 6) Compute metrics (with time filters) and plot
    metrics = calculate_metrics(cfg, res, vlbi_meta)
    plot_results(cfg, res, metrics, vlbi_meta)

    logging.info(f"Completed {mode} in {time.time() - t0:.1f}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--doy", type=int, required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    # Load YAML config
    base_cfg = parse_config()
    base_cfg.update({
        'year': args.year,
        'doy': args.doy,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'output_dir': 'experiments/'
    })

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s')

    # Define all jobs (mirrors bash)
    jobs = [
        ("GNSS", 1, 1),
        ("Fusion", 1000.0, 1),
        ("Fusion", 1, 1000.0),
        ("DTEC_Fusion", 1, 100.0),
        ("DTEC_Fusion", 100.0, 1),
    ]

    for mode, sw, lw in jobs:
        logging.info(f"Starting evaluation for {mode} with SW={sw}, LW={lw}")
        evaluate_single(base_cfg, mode, sw, lw)

    logging.info("All evaluations have completed.")

if __name__ == '__main__':
    main()
