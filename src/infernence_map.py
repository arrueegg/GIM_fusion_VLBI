import torch
import numpy as np
from spacepy.coordinates import Coords
from spacepy.time import Ticktock
from datetime import datetime, timedelta
import logging
import os

from utils.config_parser import parse_config
from models.model import get_model
from utils.data import get_data_loaders
from utils.locationencoder.pe import SphericalHarmonics

np.float_ = np.float64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def coord_transform(input_type, output_type, lats, lons, epochs):
    coords = np.array([[1 + 450 / 6371, lat, lon] for lat, lon in zip(lats, lons)], dtype=np.float64)
    geo_coords = Coords(coords, input_type, 'sph')
    geo_coords.ticks = Ticktock(epochs, 'UTC')
    return geo_coords.convert(output_type, 'sph')

def generate_grid(lats, lons, epoch):
    # Flatten latitude and longitude tensors
    lat_tensor = torch.tensor(lats.flatten())  # Shape: (71*73,)
    lon_tensor = (torch.tensor(lons.flatten()) + 180) % 360 - 180  # Shape: (71*73,)

    # Normalize latitudes and longitudes if needed
    lat_tensor = (lat_tensor - (-90)) / (90 - (-90)) * 2 - 1
    lon_tensor = (lon_tensor - (-180)) / (180 - (-180)) * 2 - 1

    # Stack longitude and latitude into a 2D tensor of shape (n_points, 2)
    lonlat_tensor = torch.stack((lon_tensor, lat_tensor), dim=-1)

    # Time-based features
    grid = {
        'sin_utc': np.sin(epoch / 86400 * 2 * np.pi),
        'cos_utc': np.cos(epoch / 86400 * 2 * np.pi),
        'sod_normalize': 2 * epoch / 86400 - 1
    }

    # Convert grid to a tensor and expand to match the number of points
    grid_tensor = torch.tensor(list(grid.values())).unsqueeze(1).T
    grid_tensor = grid_tensor.expand(lonlat_tensor.shape[0], -1)  # Shape: (71*73, 3)

    return lonlat_tensor, grid_tensor

def inference(config, model, device):
    # Generate lat/lon grid
    lat_range = np.linspace(-87.5, 87.5, 71)  # 71 lat points
    lon_range = np.linspace(-180, 180, 73)    # 73 lon points
    lats, lons = np.meshgrid(lat_range, lon_range)

    start_date = datetime.strptime(f'{config["year"]}-01-01', "%Y-%m-%d") + timedelta(days=int(config["doy"]) - 1)
    sods = np.arange(0, 87300, 900)  # Time steps

    mean_vtec_preds = []
    uncertainties = []

    for sod in sods:
        current_time = start_date + timedelta(seconds=int(sod))
        sm_coords = coord_transform('GEO', 'SM', lats.flatten(), lons.flatten(), [current_time] * len(lats.flatten()))
        lonlat, grid = generate_grid(sm_coords.lati, sm_coords.long, sod)

        # Concatenate inputs and pass to the model
        inputs = torch.cat([lonlat.to(device), grid.to(device)], dim=1).float()  # Ensure correct dtype

        model.eval()
        with torch.no_grad():
            outputs = model(inputs)

            # Extract VTEC predictions and uncertainties
            if config['training']['loss_function'] == 'LaplaceLoss':
                vtec_pred, uncertainty = outputs[:, 0], outputs[:, 1]
            else:
                vtec_pred = outputs
                uncertainty = torch.zeros_like(vtec_pred)

            # Reshape predictions and uncertainties back to (71, 73) grid
            mean_vtec_preds.append(vtec_pred.cpu().numpy().reshape(71, 73))
            uncertainties.append(uncertainty.cpu().numpy().reshape(71, 73))

    # Convert lists to 3D arrays (timesteps, lat, lon) for saving
    mean_vtec_preds = np.array(mean_vtec_preds)  # Shape: (n_timesteps, 71, 73)
    uncertainties = np.array(uncertainties)      # Shape: (n_timesteps, 71, 73)

    # Save predictions and uncertainties
    np.save(f'./experiments/maps/mean_vtec_preds_{config["year"]}_{config["doy"]}.npy', mean_vtec_preds)
    np.save(f'./experiments/maps/var_vtec_preds_{config["year"]}_{config["doy"]}.npy', uncertainties)

    return mean_vtec_preds, uncertainties

def main():
    config = parse_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #spherical_harmonics = SphericalHarmonics(legendre_polys=16)
    model = get_model(config).to(device)
    model_path = os.path.join(config['logging']['checkpoint_dir'], f'best_model_{config["model"]["model_type"]}_{config["year"]}-{config["doy"]}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    logger.info(f"Starting inference for year {config['year']} DOY {config['doy']}")
    mean_vtec, uncertainty = inference(config, model, device)

    # Save predictions
    np.save(f'./experiments/maps/mean_vtec_preds_{config["year"]}_{config["doy"]}.npy', mean_vtec)
    np.save(f'./experiments/maps/var_vtec_preds_{config["year"]}_{config["doy"]}.npy', uncertainty)
    logger.info("Inference completed.")

if __name__ == "__main__":
    main()
