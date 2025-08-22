import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta


def load_map(folder_path, approach_name, year, doy, hour):
    """Load a 2D numpy array map for the given approach, date, and hour."""

    if approach_name == "GNSS":
        lw = 1
    elif approach_name == "Fusion":
        lw = 1000
    elif approach_name == "DTEC_Fusion":
        lw = 100
    
    file_name = f"mean_vtec_preds_{year}_{doy:03d}.npy"
    subfolder = f"{approach_name}_{year}_{doy:03d}_SW1_LW{lw}/maps"
    file_path = os.path.join(folder_path, subfolder, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    map_array = np.load(file_path)

    # Ensure the hour is within bounds
    if hour < 0 or hour >= map_array.shape[0]:
        raise ValueError(f"Hour {hour} is out of bounds for the map with shape {map_array.shape}.")
    
    # Select the desired hour (assuming the first dimension corresponds to time)
    return map_array[hour]


def plot_map(map_array, title, cmap='gist_heat', coastcolors='white', vmin=None, vmax=None):
    """Plot a single 2D map with a colorbar matching the plot height."""
    # Get the current axis (should be cartopy axis)
    ax = plt.gca()
    
    # Plot the map array
    lon_range = np.linspace(-180, 180, map_array.shape[1])
    lat_range = np.linspace(-90, 90, map_array.shape[0])
    im = ax.pcolormesh(lon_range, lat_range, map_array, cmap=cmap, shading='nearest', 
                      transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
    
    # Add only coastlines (no political boundaries or other features)
    ax.coastlines(resolution='110m', linewidth=0.5, color=coastcolors)
    
    # Add latitude and longitude gridlines every 30 degrees
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,
                     xlocs=np.arange(-180, 181, 60), ylocs=np.arange(-60, 90, 30),
                     color='gray', linewidth=0.5, alpha=0.7)
    # Show labels only on left and bottom
    gl.top_labels = False
    gl.right_labels = False
    
    plt.title(title)

    # Add a colorbar
    cbar = plt.colorbar(im, fraction=0.023, pad=0.04)
    cbar.ax.tick_params(labelsize=8)


def plot_comparison(folder_path, approach1, approach2, year, doy, hour):
    """Load and plot both maps and their difference."""
    map1 = load_map(folder_path, approach1, year, doy, hour)
    map2 = load_map(folder_path, approach2, year, doy, hour)

    if map1.shape != map2.shape:
        raise ValueError("Maps must have the same shape.")
    
    date = f"{datetime.strftime(datetime(year, 1, 1) + timedelta(days=doy-1), '%d %b %Y')} at {hour}:00 UTC"

    diff_map = map1 - map2

    vmin = 0
    vmax = max(map1.max(), map2.max())

    # Plot the first map
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': ccrs.PlateCarree()})
    plot_map(map1, f"{approach1} only - {date}", vmin=vmin, vmax=vmax)
    plt.tight_layout()
    plt.savefig(f"./evaluation/compare_maps/{approach1}_only_{year}_{doy:03d}_hour{hour}.png", dpi=300, transparent=True)
    plt.close()

    # Plot the second map
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': ccrs.PlateCarree()})
    plot_map(map2, f"VLBI VTEC - {date}", vmin=vmin, vmax=vmax)
    plt.tight_layout()
    plt.savefig(f"./evaluation/compare_maps/{approach2}_VTEC_{year}_{doy:03d}_hour{hour}.png", dpi=300, transparent=True)
    plt.close()

    # Plot the difference map
    diff_vmax = max(abs(diff_map.min()), abs(diff_map.max()))
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': ccrs.PlateCarree()})
    plot_map(diff_map, f"Difference (GNSS only - VLBI VTEC)", cmap='seismic', coastcolors='black', vmin=-diff_vmax, vmax=diff_vmax)
    plt.tight_layout()
    plt.savefig(f"./evaluation/compare_maps/difference_{approach1}_{approach2}_{year}_{doy:03d}_hour{hour}.png", dpi=300, transparent=True)
    plt.close()


if __name__ == "__main__":
    folder_path = "/scratch2/arrueegg/WP2/GIM_fusion_VLBI/experiments/"
    approach1 = "GNSS"
    approach2 = "Fusion"
    year, doy, hour = 2023, 12, 20

    plot_comparison(folder_path, approach1, approach2, year, doy, hour)
