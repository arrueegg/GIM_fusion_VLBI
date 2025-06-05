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
import re
from io import StringIO
import warnings

from utils.locationencoder.pe import SphericalHarmonics
from utils.config_parser import parse_config
from models.model import get_model

warnings.filterwarnings("ignore")

# Station definitions for per-station metrics
STATION_COORDS = {
    'Kokee': (22.13, -159.66),
    'Hobart': (-42.80, 147.44),
    'Ishioka': (36.21, 140.22),
    'Santa Maria': (36.98, -25.16),
    'Onsala': (57.40, 11.92),
    'Wettzell': (49.15, 12.88),
    'Yebes': (40.52, -3.09),
    'Matera': (40.65, 16.70),
    'Sejong': (36.52, 127.30),
    'Warkworth': (-36.43, 174.66),
    'Westford': (42.61, -71.49),
    'Forteleza': (-3.88, -38.43),
    # add more stations here…
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
    year, doy = cfg['year'], cfg['doy']
    vlbi_path = cfg['data']['VLBI_data_path']
    month, day = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=doy - 1).strftime('%m %d').split()
    paths_sx = [p for p in os.listdir(os.path.join(vlbi_path, 'SX')) 
             if p.startswith(f"{year}{month}{day}-")]
    paths_vgos = [p for p in os.listdir(os.path.join(vlbi_path, 'VGOS'))
             if p.startswith(f"{year}{month}{day}-")]
    paths = set(paths_sx + paths_vgos)

    all_data = []
    for p in paths:
        fp = os.path.join(vlbi_path, p, 'summary.md')
        with open(fp, "r") as f:
            content = f.read()
        vtec_df = parse_markdown_table(content, "## VTEC Time Series")
        all_data.append(vtec_df)
    
    all_data = pd.concat(all_data, ignore_index=True)
    
    
    return vlbi_meta

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


def calculate_metrics_old(config, results):
    # Global
    rmse = round(np.sqrt(np.mean((results['model_prediction'] - results['vtec'])**2)), 2)
    mae = round(np.mean(np.abs(results['model_prediction'] - results['vtec'])), 2)
    out_dir = os.path.join(config['output_dir'], 'SA_plots')
    os.makedirs(out_dir, exist_ok=True)
    metrics_file = os.path.join(out_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"GLOBAL RMSE: {rmse}\nGLOBAL MAE: {mae}\n")

    # Per-station
    stats = []
    for station, (slat, slon) in STATION_COORDS.items():
        dists = haversine(slat, slon,
                          results['lat'].values,
                          results['lon'].values)
        subset = results[dists <= RADIUS_KM]
        if not subset.empty:
            r_s = round(np.sqrt(np.mean((subset['model_prediction'] - subset['vtec'])**2)), 2)
            m_s = round(np.mean(np.abs(subset['model_prediction'] - subset['vtec'])), 2)
            n = len(subset)
        else:
            r_s, m_s, n = None, None, 0
        stats.append({'station': station, 'RMSE': r_s, 'MAE': m_s, 'count': n})
    pd.DataFrame(stats).to_csv(os.path.join(out_dir, 'station_metrics.csv'), index=False)
    return {'RMSE': rmse, 'MAE': mae}

def calculate_metrics(config, results):
    # 1) Compute raw residuals
    results['residuals_raw'] = results['model_prediction'] - results['vtec']

    # 2) Compute global bias
    bias_global = np.mean(results['residuals_raw'])

    # 3) Compute global‐bias‐corrected residuals
    results['residuals_bc_global'] = results['residuals_raw'] - bias_global

    # 4) GLOBAL RAW METRICS
    rmse_raw = round(np.sqrt(np.mean(results['residuals_raw']**2)), 2)
    mae_raw  = round(np.mean(np.abs(results['residuals_raw'])), 2)
    std_raw  = round(np.std(results['residuals_raw']), 2)
    corr_raw = round(np.corrcoef(results['model_prediction'], results['vtec'])[0, 1], 3)

    # 5) GLOBAL-BIAS-CORRECTED METRICS
    rmse_bcg = round(np.sqrt(np.mean(results['residuals_bc_global']**2)), 2)
    mae_bcg  = round(np.mean(np.abs(results['residuals_bc_global'])), 2)
    std_bcg  = round(np.std(results['residuals_bc_global']), 2)
    # Correlation between (prediction − bias_global) and true VTEC
    corr_bcg = round(np.corrcoef(results['model_prediction'] - bias_global, results['vtec'])[0, 1], 3)

    # 6) PER-STATION METRICS
    station_stats = []
    for station, (slat, slon) in STATION_COORDS.items():
        dists = haversine(
            slat, slon,
            results['lat'].values,
            results['lon'].values
        )
        subset = results[dists <= RADIUS_KM]

        if not subset.empty:
            # 6a) Raw‐only at station
            rmse_raw_s = round(np.sqrt(np.mean(subset['residuals_raw']**2)), 2)
            mae_raw_s  = round(np.mean(np.abs(subset['residuals_raw'])), 2)

            # 6b) Global‐bias‐corrected at station
            rmse_gs = round(np.sqrt(np.mean(subset['residuals_bc_global']**2)), 2)
            mae_gs  = round(np.mean(np.abs(subset['residuals_bc_global'])),   2)

            # 6c) Local bias for this station
            local_bias = np.mean(subset['residuals_raw'])
            # Subtract local bias
            subset['residuals_bc_local'] = subset['residuals_raw'] - local_bias
            rmse_ls = round(np.sqrt(np.mean(subset['residuals_bc_local']**2)), 2)
            mae_ls  = round(np.mean(np.abs(subset['residuals_bc_local'])),      2)

            count_s = len(subset)
        else:
            rmse_raw_s, mae_raw_s = None, None
            rmse_gs, mae_gs     = None, None
            rmse_ls, mae_ls     = None, None
            count_s             = 0

        station_stats.append({
            'station':      station,
            'count':        count_s,
            'RMSE_raw':     rmse_raw_s,
            'MAE_raw':      mae_raw_s,
            'RMSE_globCor': rmse_gs,
            'MAE_globCor':  mae_gs,
            'RMSE_locCor':  rmse_ls,
            'MAE_locCor':   mae_ls
        })

    # 7) WRITE EVERYTHING INTO metrics.txt
    out_dir = os.path.join(config['output_dir'], 'SA_plots')
    os.makedirs(out_dir, exist_ok=True)
    metrics_file = os.path.join(out_dir, 'metrics.txt')

    with open(metrics_file, 'w') as f:
        # Global bias
        f.write(f"GLOBAL BIAS: {round(bias_global, 2)}\n\n")

        # Raw global metrics
        f.write("RAW GLOBAL METRICS:\n")
        f.write(f"  RMSE:        {rmse_raw}\n")
        f.write(f"  MAE:         {mae_raw}\n")
        f.write(f"  STD:         {std_raw}\n")
        f.write(f"  Correlation: {corr_raw}\n\n")

        # Global-bias-corrected global metrics
        f.write("GLOBAL-BIAS-CORRECTED GLOBAL METRICS:\n")
        f.write(f"  RMSE:        {rmse_bcg}\n")
        f.write(f"  MAE:         {mae_bcg}\n")
        f.write(f"  STD:         {std_bcg}\n")
        f.write(f"  Correlation: {corr_bcg}\n\n")

        # Per-station header
        f.write(f"PER-STATION METRICS (within {RADIUS_KM:.0f} km):\n")
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

    # 8) Return keys for plotting or further use
    return {
        'BIAS':         round(bias_global, 2),
        'RMSE_raw':     rmse_raw,  'MAE_raw':  mae_raw,  'STD_raw':  std_raw,  'CORR_raw':  corr_raw,
        'RMSE_bcg':     rmse_bcg,  'MAE_bcg':  mae_bcg,  'STD_bcg':  std_bcg,  'CORR_bcg':  corr_bcg
    }

def plot_results(config, results, metrics):
    out_dir = os.path.join(config['output_dir'], 'SA_plots')

    # 1) Predictions vs Ground Truth (raw)
    plt.figure(figsize=(8, 8))
    plt.scatter(results['model_prediction'], results['vtec'], s=0.1, alpha=0.3)
    mn, mx = results['vtec'].min(), results['vtec'].max()
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.text(
        0.05, 0.95,
        f"RMSE: {metrics['RMSE_raw']}\nMAE: {metrics['MAE_raw']}",
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
    plt.text(
        0.05, 0.95,
        f"RMSE: {metrics['RMSE_raw']}\nMAE: {metrics['MAE_raw']}",
        transform=plt.gca().transAxes, va='top',
        bbox=dict(boxstyle='round', fc='white')
    )
    plt.xlabel('Residuals (raw)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(out_dir, 'residuals_hist.png'))
    plt.close()

    # 3) Global‐bias‐corrected residual histogram
    plt.figure(figsize=(8, 8))
    # Use the global‐bias‐corrected column that calculate_metrics produced
    plt.hist(results['residuals_bc_global'], bins=100, alpha=0.75, edgecolor='black', linewidth=0.5)
    plt.text(
        0.05, 0.95,
        f"STD: {metrics['STD_bcg']}\nBias: {metrics['BIAS']}",
        transform=plt.gca().transAxes, va='top',
        bbox=dict(boxstyle='round', fc='white')
    )
    plt.xlabel('Residuals (global‐bias‐corrected)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(out_dir, 'residuals_bias_corrected_hist.png'))
    plt.close()

def evaluate_single(config, mode, sw, lw):
    # Update config for this run
    cfg = config.copy()
    cfg['mode'] = mode
    if sw is not None: cfg['vlbi_sampling_weight'] = sw
    if lw is not None: cfg['vlbi_loss_weight'] = lw
    cfg['loss_fn'] = LOSS_FN

    # Set output directory based on bash logic
    exp_root = cfg.get('output_dir')
    tag = f"SW{int(sw) if sw else 1}_LW{int(lw) if lw else 1}"
    out_dir = os.path.join(exp_root, f"{mode}_{cfg['year']}_{cfg['doy']:03d}_{tag}")
    cfg['output_dir'] = out_dir

    # Skip existing
    metrics_path = os.path.join(out_dir, 'SA_plots', 'metrics.txt')
    if os.path.exists(metrics_path) and not cfg.get('force', False):
        logging.info(f"Skipping {mode} {tag}, metrics exists")
        return

    logging.info(f"Evaluating {mode} with {tag}")
    t0 = time.time()

    vlbi_meta = load_vlbi_meta(cfg)

    # Load and filter data
    csv_path = cfg['data']['GNSS_data_path']
    if 'cluster' in csv_path:
        csv_path = '/cluster/work/igp_psr/arrueegg/sa_dataset.csv'
    sa_data = load_data(csv_path, cfg['doy'])

    # Prepare encoder and inputs
    sh_enc = None
    if cfg['preprocessing'].get('SH_encoding'):
        sh_enc = SphericalHarmonics(cfg['preprocessing']['SH_degree']).to(cfg['device'])
    inputs = prepare_inputs(sa_data, cfg['device'], sh_enc)

    # Ensemble inference
    preds = []
    for fn in os.listdir(os.path.join(out_dir, 'model')):
        m = get_model(cfg).to(cfg['device'])
        state = torch.load(os.path.join(out_dir, 'model', fn), map_location=cfg['device'])
        m.load_state_dict(state['model_state_dict'])
        m.eval()
        with torch.no_grad(): preds.append(m(inputs).cpu().numpy())
    ensemble = np.mean(preds, axis=0)

    # Save results and metrics
    res = sa_data.copy()
    res['model_prediction'] = ensemble[:,0]
    os.makedirs(os.path.join(out_dir, 'SA_plots'), exist_ok=True)
    res.to_csv(os.path.join(out_dir, 'SA_plots', 'results.csv'), index=False)

    metrics = calculate_metrics(cfg, res)
    plot_results(cfg, res, metrics)
    logging.info(f"Completed {mode} in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--doy", type=int, required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if metrics exist")
    args = parser.parse_args()

    # Load YAML config
    base_cfg = parse_config()
    base_cfg.update({
        'year': args.year,
        'doy': args.doy,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'force': args.force,
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
