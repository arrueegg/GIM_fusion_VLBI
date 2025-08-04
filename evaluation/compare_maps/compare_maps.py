import os
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
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
    # Create a Basemap instance for adding coastlines
    m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')
    m.drawcoastlines(linewidth=0.5, color=coastcolors)

    # Plot the map array
    im = plt.imshow(map_array, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, extent=(-180, 180, -90, 90))
    plt.title(title)

    # Add latitude and longitude gridlines every 30 degrees
    m.drawparallels(np.arange(-60, 90, 30), labels=[1, 0, 0, 0], fontsize=8, color='gray', linewidth=0.5)
    m.drawmeridians(np.arange(-180, 181, 60), labels=[0, 0, 0, 1], fontsize=8, color='gray', linewidth=0.5)

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
    plt.figure(figsize=(6, 3))
    plot_map(map1, f"{approach1} only - {date}", vmin=vmin, vmax=vmax)
    plt.tight_layout()
    plt.savefig(f"./evaluation/compare_maps/{approach1}_only_{year}_{doy:03d}_hour{hour}.png", dpi=300)
    plt.close()

    # Plot the second map
    plt.figure(figsize=(6, 3))
    plot_map(map2, f"VLBI VTEC - {date}", vmin=vmin, vmax=vmax)
    plt.tight_layout()
    plt.savefig(f"./evaluation/compare_maps/{approach2}_VTEC_{year}_{doy:03d}_hour{hour}.png", dpi=300)
    plt.close()

    # Plot the difference map
    diff_vmax = max(abs(diff_map.min()), abs(diff_map.max()))
    plt.figure(figsize=(6, 3))
    plot_map(diff_map, f"Difference (GNSS only - VLBI VTEC)", cmap='seismic', coastcolors='black', vmin=-diff_vmax, vmax=diff_vmax)
    plt.tight_layout()
    plt.savefig(f"./evaluation/compare_maps/difference_{approach1}_{approach2}_{year}_{doy:03d}_hour{hour}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    folder_path = "/scratch2/arrueegg/WP2/GIM_fusion_VLBI/experiments/"
    approach1 = "GNSS"
    approach2 = "Fusion"
    year, doy, hour = 2023, 12, 20

    plot_comparison(folder_path, approach1, approach2, year, doy, hour)
