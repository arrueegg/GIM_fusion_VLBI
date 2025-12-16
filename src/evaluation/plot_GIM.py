###############################
# Author: Arno Rüegg
# Date: 2024-11-29
# Description: Plot GIMs (VTEC and RMS)
###############################

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import datetime
import argparse
from PIL import Image
import glob

def load_GIM(args):
    # Construct the filename based on DOY and year
    year = args.year
    doy = args.doy
    GIM_path = args.GIM_path

    # Last two digits of the year
    yy = str(year)[-2:]

    # Day of year padded to 3 digits
    doy_str = f"{doy:03d}"

    # You can change 'IGS' to the appropriate prefix if needed
    filename = f"igsg{doy_str}0.{yy}i"
    filepath = os.path.join(GIM_path, str(year), filename)

    # Check if file exists
    if not os.path.isfile(filepath):
        print(f"File {filepath} does not exist.")
        return None

    # Parse the IONEX file
    gim_data = parse_ionex(filepath)

    return gim_data

def parse_ionex(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Initialize variables
    lat_start = lat_end = lat_step = None
    lon_start = lon_end = lon_step = None
    epochs = []

    # Parse the header
    header = True
    for i, line in enumerate(lines):
        if 'END OF HEADER' in line:
            header = False
            header_end_line = i
            break
        elif 'LAT1 / LAT2 / DLAT' in line:
            line_splitted = line.strip().split()
            lat_start = float(line_splitted[0])
            lat_end = float(line_splitted[1])
            lat_step = float(line_splitted[2])
        elif 'LON1 / LON2 / DLON' in line:
            line_splitted = line.strip().split()
            lon_start = float(line_splitted[0])
            lon_end = float(line_splitted[1])
            lon_step = float(line_splitted[2])

    if header:
        print("No END OF HEADER found.")
        return None

    # Generate latitude and longitude arrays
    lats = np.arange(lat_start, lat_end + lat_step/2, lat_step)
    lons = np.arange(lon_start, lon_end + lon_step/2, lon_step)

    # Initialize data structures for all epochs
    vtec_maps = []
    rms_maps = []
    epochs = []

    # Now parse the data
    i = header_end_line + 1
    while i < len(lines):
        line = lines[i]
        if 'START OF TEC MAP' in line:
            # Read the epoch
            i += 1
            while 'EPOCH OF CURRENT MAP' not in lines[i]:
                i += 1
            epoch_line = lines[i]
            # Extract epoch
            epoch_values = epoch_line.strip().split()
            # Year, month, day, hour, minute, second
            year = int(epoch_values[0])
            month = int(epoch_values[1])
            day = int(epoch_values[2])
            hour = int(epoch_values[3])
            minute = int(epoch_values[4])
            second = float(epoch_values[5])
            epochs.append((year, month, day, hour, minute, second))

            # Initialize VTEC map for this epoch
            vtec_map = np.zeros((len(lats), len(lons)))

            # Read the VTEC map
            i += 1
            while i < len(lines):
                line = lines[i]
                if 'END OF TEC MAP' in line:
                    break
                elif 'LAT/LON1/LON2/DLON' in line:
                    line_splitted = line.replace('-', ' -').strip().split()
                    # Read coordinates
                    lat = float(line_splitted[0])
                    lon1 = float(line_splitted[1])
                    lon2 = float(line_splitted[2])
                    dlon = float(line_splitted[3])

                    # Read the data lines
                    n_lons = int(round((lon2 - lon1) / dlon)) + 1
                    n_values_read = 0
                    values = []
                    while n_values_read < n_lons:
                        i += 1
                        data_line = lines[i]
                        data_line = data_line.strip()
                        data_values = [float(vtec)/10 for vtec in data_line.split()]
                        values.extend(data_values)
                        n_values_read += len(data_values)
                    # Store the values in the vtec_map
                    lat_idx = np.where(np.isclose(lats, lat))[0][0]
                    vtec_map[lat_idx, :] = values
                i += 1
            vtec_maps.append(vtec_map)
            i += 1

        elif 'START OF RMS MAP' in line:
            # Initialize RMS map for this epoch
            rms_map = np.zeros((len(lats), len(lons)))
            # Read the RMS map
            i += 1
            while i < len(lines):
                line = lines[i]
                if 'END OF RMS MAP' in line:
                    break
                elif 'LAT/LON1/LON2/DLON' in line:
                    line_splitted = line.replace('-', ' -').strip().split()
                    # Read coordinates
                    lat = float(line_splitted[0])
                    lon1 = float(line_splitted[1])
                    lon2 = float(line_splitted[2])
                    dlon = float(line_splitted[3])

                    # Read the data lines
                    n_lons = int(round((lon2 - lon1) / dlon)) + 1
                    n_values_read = 0
                    values = []
                    while n_values_read < n_lons:
                        i += 1
                        data_line = lines[i]
                        data_line = data_line.strip()
                        data_values = [float(rms)/10 for rms in data_line.split()]
                        values.extend(data_values)
                        n_values_read += len(data_values)
                    # Store the values in the rms_map
                    lat_idx = np.where(np.isclose(lats, lat))[0][0]
                    rms_map[lat_idx, :] = values
                i += 1
            rms_maps.append(rms_map)
            i += 1
        else:
            i += 1

    # Convert lists to numpy arrays
    vtec_maps = np.array(vtec_maps)  # Shape: (num_epochs, num_lats, num_lons)
    rms_maps = np.array(rms_maps)

    # Return the data
    gim_data = {
        'lats': lats,
        'lons': lons,
        'epochs': epochs,
        'vtec': vtec_maps,
        'rms': rms_maps
    }
    return gim_data

def plot_GIM(gim, args):
    vtec_maps = gim['vtec']
    rms_maps = gim['rms']

    # Create a meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(gim['lons'], gim['lats'])

    date = datetime.datetime(*map(int, gim['epochs'][0][:6])).strftime('%Y%m%d')
    
    # List to store image paths for GIF creation
    image_paths = []

    for epoch in range(len(gim['epochs'])):

        vtec = vtec_maps[epoch]
        rms = rms_maps[epoch]

        # Extract and format the date from the tuple
        epoch_tuple = gim['epochs'][epoch]
        epoch_datetime = datetime.datetime(*map(int, epoch_tuple[:6]))

        # Plot VTEC and RMS
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        plt.suptitle('IGS GIM for ' + epoch_datetime.strftime('%Y-%m-%d %H:%M'), fontweight='bold', fontsize=16)

        # VTEC Plot
        ax1 = axes[0]
        vtec_plot = ax1.pcolormesh(lon_grid, lat_grid, vtec, shading='auto', cmap='gist_heat', vmin=0, vmax=80)
        ax1.coastlines(color='white')
        ax1.set_title('VTEC Map', fontsize=14)
        ax1.set_xlabel('Longitude', fontsize=12)
        ax1.set_ylabel('Latitude', fontsize=12)
        ax1.set_aspect('equal') 
        ax1.set_xticks(np.arange(-180, 181, 60))
        ax1.set_yticks(np.arange(-90, 91, 30))
        ax1.tick_params(labelsize=11)
        ax1.grid(True, alpha=0.3)
        cbar1 = fig.colorbar(vtec_plot, ax=ax1, label='VTEC (TECU)')
        cbar1.ax.tick_params(labelsize=11)
        cbar1.set_label('VTEC (TECU)', fontsize=12)

        # RMS Plot
        ax2 = axes[1]
        rms_plot = ax2.pcolormesh(lon_grid, lat_grid, rms, shading='auto', cmap='Blues', vmin=0, vmax=20)
        ax2.coastlines(color='black')
        ax2.set_title('RMS Map', fontsize=14)
        ax2.set_xlabel('Longitude', fontsize=12)
        ax2.set_ylabel('Latitude', fontsize=12)
        ax2.set_aspect('equal')
        ax2.set_xticks(np.arange(-180, 181, 60))
        ax2.set_yticks(np.arange(-90, 91, 30))
        ax2.tick_params(labelsize=11)
        ax2.grid(True, alpha=0.3)
        cbar2 = fig.colorbar(rms_plot, ax=ax2, label='RMS (TECU)')
        cbar2.ax.tick_params(labelsize=11)
        cbar2.set_label('RMS (TECU)', fontsize=12)

        plt.tight_layout()
        os.makedirs(f'evaluation/IGS_GIM/{date}', exist_ok=True)
        
        # Save the image and store path for GIF
        image_path = f'evaluation/IGS_GIM/{date}/GIM_{epoch_datetime.strftime("%Y%m%d_%H%M")}.png'
        plt.savefig(image_path)
        image_paths.append(image_path)
        
        plt.show()
        plt.close()
    
    # Create GIF from all daily images
    create_gif(image_paths, f'evaluation/IGS_GIM/{date}/GIM_daily_{date}.gif')

def plot_VTEC_only(gim, args):
    vtec_maps = gim['vtec']

    # Create a meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(gim['lons'], gim['lats'])

    date = datetime.datetime(*map(int, gim['epochs'][0][:6])).strftime('%Y%m%d')
    
    # List to store image paths for GIF creation
    image_paths = []

    for epoch in range(len(gim['epochs'])):

        vtec = vtec_maps[epoch]

        # Extract and format the date from the tuple
        epoch_tuple = gim['epochs'][epoch]
        epoch_datetime = datetime.datetime(*map(int, epoch_tuple[:6]))

        # Plot VTEC only
        fig, ax = plt.subplots(1, 1, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

        # VTEC Plot
        vtec_plot = ax.pcolormesh(lon_grid, lat_grid, vtec, shading='auto', cmap='gist_heat', vmin=0, vmax=80)
        ax.coastlines(color='white')
        ax.set_title('IGS GIM VTEC for ' + epoch_datetime.strftime('%Y-%m-%d %H:%M'), fontweight='bold', fontsize=16)
        ax.set_xlabel('Longitude', fontsize=14)
        ax.set_ylabel('Latitude', fontsize=14)
        ax.set_aspect('equal') 
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.3)
        
        # Create colorbar using the simpler approach
        cbar = fig.colorbar(vtec_plot, ax=ax, label='VTEC (TECU)', shrink=0.8)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('VTEC (TECU)', fontsize=14)

        plt.tight_layout()
        os.makedirs(f'evaluation/IGS_GIM/{date}', exist_ok=True)
        
        # Save the image and store path for GIF
        image_path = f'evaluation/IGS_GIM/{date}/GIM_VTEC_only_{epoch_datetime.strftime("%Y%m%d_%H%M")}.png'
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        image_paths.append(image_path)
        
        plt.show()
        plt.close()
    
    # Create GIF from all daily images
    create_gif(image_paths, f'evaluation/IGS_GIM/{date}/GIM_VTEC_daily_{date}.gif')

def create_gif(image_paths, output_path, duration=500):
    """
    Create a GIF from a list of image paths.
    
    Parameters:
    image_paths (list): List of paths to PNG images
    output_path (str): Path for the output GIF file
    duration (int): Duration between frames in milliseconds
    """
    if not image_paths:
        print("No images found to create GIF")
        return
    
    # Load all images
    images = []
    for path in image_paths:
        if os.path.exists(path):
            img = Image.open(path)
            images.append(img)
    
    if images:
        # Save as GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF created: {output_path}")
    else:
        print("No valid images found to create GIF")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2024,help='year of data to process')
    parser.add_argument('--doy', type=int, default=183, help='day of year of data to process')
    parser.add_argument('--GIM_path', type=str, default='/home/space/project/2022_shumao_IonoSpatialModeling/07_data/GNSS_ionex/', help='Path to the GIM folder')
    parser.add_argument('--vtec_only', action='store_true', default=True, help='Plot only VTEC without RMS subplot')
    args = parser.parse_args()

    # Load GIM
    gim = load_GIM(args)

    # Plot VTEC and RMS or VTEC only based on flag
    if args.vtec_only:
        plot_VTEC_only(gim, args)
    else:
        plot_GIM(gim, args)

if __name__ == "__main__":
    main()