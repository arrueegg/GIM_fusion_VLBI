import torch
import numpy as np
from spacepy.coordinates import Coords
from spacepy.time import Ticktock
from datetime import datetime, timedelta
import logging
import os
import re

from utils.config_parser import parse_config
from models.model import get_model
from utils.data import get_data_loaders

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from utils.locationencoder.pe import SphericalHarmonics

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.float_ = np.float64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def coord_transform(input_type, output_type, lats, lons, epochs):
    coords = np.array([[1 + 450 / 6371, lat, lon] for lat, lon in zip(lats, lons)], dtype=np.float64)
    geo_coords = Coords(coords, input_type, 'sph')
    geo_coords.ticks = Ticktock(epochs, 'UTC')
    return geo_coords.convert(output_type, 'sph')

def generate_grid(config, lat_dim, lon_dim, sod, date):
    # Step 1: Define latitude and longitude ranges in GEO
    lat_range = np.linspace(-87.5, 87.5, lat_dim)
    lon_range = np.linspace(-180, 180, lon_dim)
    lats, lons = np.meshgrid(lat_range, lon_range, indexing='ij')
    
    # Flatten for transformation (required by coord_transform)
    flat_lats = lats.flatten()
    flat_lons = lons.flatten()
    epochs = [date + timedelta(seconds=int(sod))] * len(flat_lats)
    
    # Step 2: Transform to SM coordinates
    sm_coords = coord_transform('GEO', 'SM', flat_lats, flat_lons, epochs)
    sm_lats, sm_lons = sm_coords.lati, sm_coords.long
    
    # Step 3: Normalize SM latitude and longitude to [-1, 1]
    lat_tensor = torch.tensor(sm_lats, dtype=torch.float32)
    lon_tensor = torch.tensor(sm_lons, dtype=torch.float32)
    if not config['preprocessing']['SH_encoding']:
        lat_tensor = (lat_tensor + 90) / 180 * 2 - 1
        lon_tensor = (lon_tensor + 180) / 360 * 2 - 1
    
    # Combine normalized coordinates into a 2D tensor
    lonlat_tensor = torch.stack((lon_tensor, lat_tensor), dim=-1)

    if config['preprocessing']['SH_encoding']:
        sh_encoder = SphericalHarmonics(legendre_polys=config['preprocessing']['SH_degree'])
        lonlat_tensor = sh_encoder(lonlat_tensor)
    
    # Step 4: Time-based features for the given `sod`
    sin_utc = np.sin(sod / 86400 * 2 * np.pi)
    cos_utc = np.cos(sod / 86400 * 2 * np.pi)
    sod_normalized = 2 * sod / 86400 - 1

    time_tensor = torch.tensor([sin_utc, cos_utc, sod_normalized], dtype=torch.float32).expand(lonlat_tensor.shape[0], -1)
    
    return lonlat_tensor, time_tensor

def inference(config, model, device, lat_dim=71, lon_dim=73, interval=3600):

    date = datetime.strptime(f"{config['year']}-01-01", "%Y-%m-%d") + timedelta(days=int(config['doy']) - 1)
    sods = np.arange(0, 86400  + interval, interval)  # Time steps

    mean_vtec_preds = []
    uncertainties = []

    for sod in sods:
        lonlat, time_features = generate_grid(config, lat_dim, lon_dim, sod, date)

        # Concatenate inputs and pass to the model
        inputs = torch.cat([lonlat.to(device), time_features.to(device)], dim=1).float()

        model.eval()
        with torch.no_grad():
            outputs = model(inputs)

            # Extract VTEC predictions and uncertainties
            if config['training']['loss_function'] == 'LaplaceLoss' or config['training']['loss_function'] == 'GaussianNLLLoss':
                vtec_pred, uncertainty = outputs[:, 0], outputs[:, 1]
            else:
                vtec_pred = outputs
                uncertainty = torch.zeros_like(vtec_pred)

            # Reshape predictions and uncertainties back to (71, 73) grid
            mean_vtec_preds.append(vtec_pred.cpu().numpy().reshape(lat_dim, lon_dim))
            uncertainties.append(uncertainty.cpu().numpy().reshape(lat_dim, lon_dim))

    # Convert lists to 3D arrays (timesteps, lat, lon) for saving
    mean_vtec_preds = np.array(mean_vtec_preds)  
    uncertainties = np.array(uncertainties)     

    # Save predictions and uncertainties
    os.makedirs(f"{config['output_dir']}/maps", exist_ok=True)
    np.save(f"{config['output_dir']}/maps/mean_vtec_preds_{config['year']}_{config['doy']}.npy", mean_vtec_preds)
    np.save(f"{config['output_dir']}/maps/var_vtec_preds_{config['year']}_{config['doy']}.npy", uncertainties)

    return mean_vtec_preds, uncertainties

def plot_mean(config, vtec_data, std_data, lat_dim, lon_dim):

    vtec_data = np.mean(vtec_data, axis=0)
    std_data = np.mean(std_data, axis=0)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Latitude and longitude ranges matching the VTEC data
    lat_range = np.linspace(-87.5, 87.5, lat_dim)   # 71 lat points
    lon_range = np.linspace(-180, 180, lon_dim)     # 73 lon points

    # Plot VTEC data
    ax1.set_title(f"VTEC Map {config['year']} {config['doy']}")
    vtec_plot = ax1.pcolormesh(lon_range, lat_range, vtec_data, cmap='viridis', shading='nearest', transform=ccrs.PlateCarree(), vmin=0, vmax=80)
    ax1.coastlines()
    fig.colorbar(vtec_plot, ax=ax1, orientation='vertical', label='VTEC')

    # Plot standard deviation data
    ax2.set_title(f"Standard Deviation Map {config['year']} {config['doy']}")
    std_plot = ax2.pcolormesh(lon_range, lat_range, std_data, cmap='Blues', shading='nearest', transform=ccrs.PlateCarree(), vmin=0, vmax=20)
    ax2.coastlines()
    fig.colorbar(std_plot, ax=ax2, orientation='vertical', label='Standard Deviation')

    # Show the plots
    plt.tight_layout()
    os.makedirs(f"{config['output_dir']}/plots/{config['year']}_{config['doy']}", exist_ok=True)
    plt.savefig(f"{config['output_dir']}/plots/{config['year']}_{config['doy']}/mean_map_{config['year']}_{config['doy']}_{config['model']['model_type']}_{config['training']['loss_function']}_lossW{config['training']['vlbi_loss_weight']}_samplingW{config['training']['vlbi_sampling_weight']}.png")
    #plt.show()
    plt.close()

def plot_epoch(config, vtec_data, std_data, lat_dim, lon_dim, interval):

    sods = np.arange(0, 86400  + interval, interval)

    for i in range(vtec_data.shape[0]):
    
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

        # Latitude and longitude ranges matching the VTEC data
        lat_range = np.linspace(-87.5, 87.5, lat_dim)   # 71 lat points
        lon_range = np.linspace(-180, 180, lon_dim)     # 73 lon points
        time = f"{(sods[i] // 3600):02d}:{(sods[i] % 3600 // 60):02d}"

        # Plot VTEC data
        ax1.set_title(f"VTEC Map Epoch {config['year']}-{config['doy']}-{time}")
        vtec_plot = ax1.pcolormesh(lon_range, lat_range, vtec_data[i], shading='nearest', cmap='viridis', transform=ccrs.PlateCarree(), vmin=0, vmax=80)
        ax1.coastlines()
        fig.colorbar(vtec_plot, ax=ax1, orientation='vertical', label='VTEC')

        # Plot standard deviation data
        ax2.set_title(f"Standard Deviation Map Epoch {config['year']}-{config['doy']}-{time}")
        std_plot = ax2.pcolormesh(lon_range, lat_range, std_data[i], shading='nearest', cmap='Blues', transform=ccrs.PlateCarree(), vmin=0, vmax=20)
        ax2.coastlines()
        fig.colorbar(std_plot, ax=ax2, orientation='vertical', label='Standard Deviation')

        # Show the plots
        plt.tight_layout()
        os.makedirs(f"{config['output_dir']}/plots/{config['year']}_{config['doy']}", exist_ok=True)
        plt.savefig(f"{config['output_dir']}/plots/{config['year']}_{config['doy']}/VTEC_and_STD_{config['year']}_{config['doy']}_{sods[i]}_{config['model']['model_type']}_{config['training']['loss_function']}_lossW{config['training']['vlbi_loss_weight']}_samplingW{config['training']['vlbi_sampling_weight']}.png")
        #plt.show()
        plt.close()

def create_gif_from_images(config):
    # Path to the folder containing the images
    folder_path = f"{config['output_dir']}/plots/{config['year']}_{config['doy']}/"

    # Function to extract the numeric part from the filename
    def extract_number(filename):
        match = re.search(r'_(\d+)_MLP', filename)
        return int(match.group(1)) if match else 0

    # Collect all image file paths, ensuring they are sorted numerically
    image_files = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png') and not f.startswith('mean')],
        key=extract_number
    )

    # Load images and create the GIF
    images = [imageio.imread(file) for file in image_files]
    gif_path = f"{folder_path}/VTEC_and_STD_{config['year']}_{config['doy']}_{config['model']['model_type']}_{config['training']['loss_function']}_lossW{config['training']['vlbi_loss_weight']}_samplingW{config['training']['vlbi_sampling_weight']}.gif"

    # Save the images as a GIF
    imageio.mimsave(
        gif_path,
        images,
        duration=0.5,  # Duration between frames (adjust as needed)
        palettesize=256,  # Maximum number of colors (256 is the max for GIFs)
        subrectangles=True,  # Compress repeating pixels to reduce file size
        loop=0
    )
    
def main():

    # Generate lat/lon grid dimensions
    lat_dim = int((175 // 1) + 1) 
    lon_dim = int((360 // 1) + 1) 
    interval = 3600 #900 (15min)

    config = parse_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #spherical_harmonics = SphericalHarmonics(legendre_polys=16)
    logger.info(f"Starting inference for year {config['year']} DOY {config['doy']}")

    model = get_model(config).to(device)
    model_path = os.path.join(config['output_dir'], 'model', f"best_model_{config['data']['mode']}_{config['model']['model_type']}_{config['year']}-{config['doy']}.pth")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)['model_state_dict'])
    vtec, uncertainty = inference(config, model, device, lat_dim, lon_dim, interval)

    # Save predictions
    os.makedirs(f"{config['output_dir']}/maps", exist_ok=True)
    np.save(f"{config['output_dir']}/maps/mean_vtec_preds_{config['year']}_{config['doy']}.npy", vtec)
    np.save(f"{config['output_dir']}/maps/var_vtec_preds_{config['year']}_{config['doy']}.npy", uncertainty)
    logger.info("Inference completed.")

    plot_mean(config, vtec, uncertainty, lat_dim, lon_dim)
    plot_epoch(config, vtec, uncertainty, lat_dim, lon_dim, interval)
    #plot_epoch_animation(config, vtec, uncertainty, lat_dim, lon_dim, interval)

    create_gif_from_images(config)
    logger.info("Plots and GIF creation completed.")

if __name__ == "__main__":
    main()
