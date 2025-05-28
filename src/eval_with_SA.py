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
import warnings

from utils.locationencoder.pe import SphericalHarmonics
from utils.config_parser import parse_config
from models.model import get_model

warnings.filterwarnings("ignore")

# Station definitions for per-station metrics
STATION_COORDS = {
    'Kokee': (22.14, -159.64),
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


def load_data(csv_file, doy):
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    return df[df['time'].dt.dayofyear == doy]


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


def calculate_metrics(config, results):
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


def plot_results(config, results, metrics):
    out_dir = os.path.join(config['output_dir'], 'SA_plots')
    # Predictions vs Ground Truth
    plt.figure(figsize=(8,8))
    plt.scatter(results['model_prediction'], results['vtec'], s=0.1, alpha=0.3)
    mn, mx = results['vtec'].min(), results['vtec'].max()
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.text(0.05, 0.95, f"RMSE: {metrics['RMSE']}\nMAE: {metrics['MAE']}",
             transform=plt.gca().transAxes, va='top', bbox=dict(boxstyle='round', fc='white'))
    plt.xlabel('Model Prediction')
    plt.ylabel('VTEC (Ground Truth)')
    plt.savefig(os.path.join(out_dir, 'pred_vs_gt.png'))
    plt.close()

    # Residual histogram
    plt.figure(figsize=(8,8))
    residuals = results['model_prediction'] - results['vtec']
    plt.hist(residuals, bins=100, alpha=0.75)
    plt.text(0.05, 0.95, f"RMSE: {metrics['RMSE']}\nMAE: {metrics['MAE']}",
             transform=plt.gca().transAxes, va='top', bbox=dict(boxstyle='round', fc='white'))
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(out_dir, 'residuals_hist.png'))
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

    # Load and filter data
    #csv_path = cfg['data']['GNSS_data_path']
    csv_path = '/home/space/internal/ggltmp/4Arno/sa_dataset.csv'
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
