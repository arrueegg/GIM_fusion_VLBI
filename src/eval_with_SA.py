from math import e
import pandas as pd
import numpy as np
import torch
import os
import logging
import matplotlib.pyplot as plt
from utils.locationencoder.pe import SphericalHarmonics
from utils.config_parser import parse_config
from models.model import get_model
import warnings
warnings.filterwarnings("ignore")

def load_data(csv_file, doy):
    """
    Load satellite altimetry data for a specific day of the year (DOY).

    Parameters:
    csv_file (str): Path to the CSV file containing the data.
    doy (int): Day of the year to filter the data.

    Returns:
    pd.DataFrame: Filtered DataFrame containing data for the specified DOY.
    """
    data = pd.read_csv(csv_file)
    data['time'] = pd.to_datetime(data['time'])
    data = data[data['time'].dt.dayofyear == doy]
    return data

def prepare_inputs(sa_data, device, sh_encoder):
    """
    Prepare inputs for the model by encoding lat/lon and adding time-based features.

    Parameters:
    sa_data (pd.DataFrame): DataFrame containing the satellite altimetry data.
    device (torch.device): Device to move tensors to.
    sh_encoder (SphericalHarmonics): Instance for spherical harmonics encoding.

    Returns:
    torch.Tensor: Prepared input tensor for the model.
    """
    lat_tensor = torch.tensor(sa_data['sm_lat'].values, dtype=torch.float32).to(device)
    lon_tensor = torch.tensor(sa_data['sm_lon'].values, dtype=torch.float32).to(device)
    inputs = torch.stack([lon_tensor, lat_tensor], dim=1).float()

    if sh_encoder:
        inputs = sh_encoder(inputs)

    # Calculate sod (Seconds of Day)
    sod = torch.tensor((sa_data['time'].dt.hour * 3600 + sa_data['time'].dt.minute * 60 + sa_data['time'].dt.second).values, dtype=torch.float32).to(device)
    sin_utc = torch.sin(sod / 86400 * 2 * torch.pi)
    cos_utc = torch.cos(sod / 86400 * 2 * torch.pi)
    sod_normalized = 2 * sod / 86400 - 1

    inputs = torch.cat([inputs, sin_utc.unsqueeze(1), cos_utc.unsqueeze(1), sod_normalized.unsqueeze(1)], dim=1)
    return inputs

def evaluate_results(sa_data, predictions, config):
    """
    Compare model predictions to ground truth from the SA dataset.

    Parameters:
    sa_data (pd.DataFrame): DataFrame containing the satellite altimetry data.
    predictions (np.ndarray): Model predictions for the SA data.

    Returns:
    pd.DataFrame: DataFrame containing ground truth and model predictions.
    """
    results = sa_data.copy()
    results['model_prediction'] = predictions[:, 0]
    results['uncertainty'] = predictions[:, 1] if predictions.shape[1] > 1 else np.nan
    out_path = os.path.join(config['output_dir'], 'SA_plots')
    os.makedirs(out_path, exist_ok=True)
    results.to_csv(os.path.join(out_path, "results.csv"), index=False)
    return results

def plot_results(config, results, metrics):
    """
    Generate plots to visualize model predictions versus ground truth.

    Parameters:
    results (pd.DataFrame): DataFrame containing ground truth and model predictions.
    """
    out_path = os.path.join(config['output_dir'], 'SA_plots')
    os.makedirs(out_path, exist_ok=True)

    # Plotting model predictions vs ground truth
    plt.figure(figsize=(8, 8))
    plt.scatter(results['model_prediction'], results['vtec'], alpha=0.3, s=0.1, label='Predictions vs Ground Truth')
    plt.xlabel('Model Prediction')
    plt.ylabel('Ground Truth (VTEC)')
    plt.plot([results['vtec'].min(), results['vtec'].max()], [results['vtec'].min(), results['vtec'].max()], 'r--', lw=1, label='Ideal Fit')
    plt.title('Model Predictions vs Ground Truth')
    plt.text(0.05, 0.95, f"RMSE: {metrics['RMSE']}\nMAE: {metrics['MAE']}", 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(os.path.join(out_path, "predictions_vs_ground_truth.png"))
    plt.close()

    # Plotting residuals as histogram
    plt.figure(figsize=(8, 8))
    residuals = results['model_prediction'] - results['vtec']
    plt.hist(residuals, bins=100, alpha=0.75, label='Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Histogram (Model Pred. - GT SA)')
    plt.text(0.05, 0.95, f"RMSE: {metrics['RMSE']}\nMAE: {metrics['MAE']}", 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_path, "residuals_histogram.png"))
    plt.close()
    

def calculate_metrics(results):
    """
    Calculate evaluation metrics for model predictions.

    Parameters:
    results (pd.DataFrame): DataFrame containing ground truth and model predictions.

    Returns:
    dict: Dictionary containing RMSE and mean absolute error.
    """
    rmse = np.sqrt(np.mean((results['model_prediction'] - results['vtec']) ** 2))
    mae = np.mean(np.abs(results['model_prediction'] - results['vtec']))
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2)}

def main():
    """
    Main function to perform forward pass, evaluation, and visualization.
    """
    config = parse_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configure logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    csv_file = '/home/ggl/mnt/ggltmp/4Arno/sa_dataset.csv'
    doy = int(config['doy'])

    logger.info(f"Loading data for DOY {doy}")
    sa_data = load_data(csv_file, doy)

    sh_encoder = None
    if config['preprocessing']['SH_encoding']:
        sh_encoder = SphericalHarmonics(legendre_polys=config['preprocessing']['SH_degree']).to(device)

    logger.info("Preparing inputs")
    inputs = prepare_inputs(sa_data, device, sh_encoder)

    ensemble_predictions = []

    logger.info("Inference...")
    model_paths = os.listdir(os.path.join(config['output_dir'], 'model'))
    for model_path in model_paths:
        model_path = os.path.join(config['output_dir'], 'model', model_path)
        model = get_model(config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])

        model.eval()
        with torch.no_grad():
            predictions = model(inputs).cpu().numpy()
        ensemble_predictions.append(predictions)

    logger.info("Aggregating ensemble predictions")
    ensemble_predictions = np.array(ensemble_predictions)
    mean_predictions = np.mean(ensemble_predictions, axis=0)
    
    results = evaluate_results(sa_data, mean_predictions, config)

    logger.info("Calculating metrics")
    metrics = calculate_metrics(results)
    logger.info(f"Metrics: {metrics}")

    logger.info("Generating plots")
    plot_results(config, results, metrics)

    logger.info("Evaluation complete")

if __name__ == "__main__":
    main()
