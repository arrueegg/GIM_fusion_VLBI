import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import argparse
import os

def parser():
    parser = argparse.ArgumentParser(description="plot Maps")
    parser.add_argument("--year", type=str, default="2023")
    parser.add_argument("--doy", type=str, default="001")
    parser.add_argument("--map_path", type=str, default="./experiments/maps/")
    parser.add_argument("--debug", type=str, default="False")
    args = parser.parse_args()
    return args

def load_maps(args):
    vtec_path = os.path.join(args.map_path, f"mean_vtec_preds_{args.year}_{args.doy}.npy")
    std_path = os.path.join(args.map_path, f"var_vtec_preds_{args.year}_{args.doy}.npy")
    vtec_data = np.load(vtec_path)
    std_data = np.load(std_path)

    """n_timesteps = vtec_data.shape[0]  # Number of timestamps
    vtec_data = vtec_data.reshape(n_timesteps, 71, 73)
    std_data = std_data.reshape(n_timesteps, 71, 73)"""

    return vtec_data, std_data

def plot_mean(vtec_data, std_data):

    vtec_data = np.mean(vtec_data, axis=0)
    #vtec_data = vtec_data[1]
    std_data = np.mean(std_data, axis=0)
    #std_data = std_data[1]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Latitude and longitude ranges matching the VTEC data
    lat_range = np.linspace(-87.5, 87.5, 71)   # 71 lat points
    lon_range = np.linspace(-180, 180, 73)     # 73 lon points

    # Plot VTEC data
    ax1.set_title('VTEC Map')
    vtec_plot = ax1.pcolormesh(lon_range, lat_range, vtec_data, cmap='viridis', shading='nearest', transform=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    fig.colorbar(vtec_plot, ax=ax1, orientation='vertical', label='VTEC')

    # Plot standard deviation data
    ax2.set_title('Standard Deviation Map')
    std_plot = ax2.pcolormesh(lon_range, lat_range, std_data, cmap='viridis', shading='nearest', transform=ccrs.PlateCarree())
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    fig.colorbar(std_plot, ax=ax2, orientation='vertical', label='Standard Deviation')

    # Show the plots
    plt.tight_layout()
    plt.show()

def plot_epoch(vtec_data, std_data):

    for i in range(vtec_data.shape[0]):
    
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

        # Latitude and longitude ranges matching the VTEC data
        lat_range = np.linspace(-87.5, 87.5, 71)   # 71 lat points
        lon_range = np.linspace(-180, 180, 73)     # 73 lon points

        # Plot VTEC data
        ax1.set_title('VTEC Map Epoch ' + str(i))
        vtec_plot = ax1.pcolormesh(lon_range, lat_range, vtec_data[i], shading='nearest', cmap='viridis', transform=ccrs.PlateCarree())
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS, linestyle=':')
        fig.colorbar(vtec_plot, ax=ax1, orientation='vertical', label='VTEC')

        # Plot standard deviation data
        ax2.set_title('Standard Deviation Map Epoch ' + str(i))
        std_plot = ax2.pcolormesh(lon_range, lat_range, std_data[i], shading='nearest', cmap='viridis', transform=ccrs.PlateCarree())
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS, linestyle=':')
        fig.colorbar(std_plot, ax=ax2, orientation='vertical', label='Standard Deviation')

        # Show the plots
        plt.tight_layout()
        os.makedirs("experiments/plots/", exist_ok=True)
        plt.savefig(f"experiments/plots/VTEC_and_STD_{i}.png")
        plt.show()

def main():
    args = parser()

    vtec_data, std_data = load_maps(args)
    print(vtec_data.shape)
    plot_mean(vtec_data, std_data)
    plot_epoch(vtec_data, std_data)

if __name__ == "__main__":
    main()