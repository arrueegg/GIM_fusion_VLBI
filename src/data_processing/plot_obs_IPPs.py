import numpy as np
import pandas as pd
import h5py
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def read_vtec_data(folder, year, doy):
    """
    Reads the VTEC data file and returns it as a pandas DataFrame,
    selecting specific columns from the 'all_data' structured dataset.
    """
    file_path = f"{folder}/{year}/{doy:03d}/ccl_{year}{doy:03d}_30_5.h5"

    try:
        with h5py.File(file_path, 'r') as h5_file:
            # Navigate to the 'all_data' dataset
            # The path to 'all_data' should be relative to the H5 file root.
            # Based on your description: year/doy/all_data
            all_data_path = f"{year}/{doy:03d}/all_data"
            
            if all_data_path not in h5_file:
                print(f"Error: Dataset '{all_data_path}' not found in {file_path}")
                return pd.DataFrame() # Return empty DataFrame or raise an error

            all_data_dataset = h5_file[all_data_path]

            # Define the columns you want to read
            selected_columns = [
                'station', 'sat', 'stec', 'vtec', 'satele', 'satazi',
                'lon_ipp', 'lat_ipp', 'sod', 'lat_sta', 'lon_sta'
            ]

            # Check if all selected columns exist in the dataset's dtype
            dataset_fields = all_data_dataset.dtype.names
            if not dataset_fields: # Check if it's a structured array at all
                 print(f"Error: '{all_data_path}' is not a structured array or has no fields.")
                 return pd.DataFrame()

            for col in selected_columns:
                if col not in dataset_fields:
                    print(f"Warning: Column '{col}' not found in dataset '{all_data_path}'. Available fields: {dataset_fields}")
                    # You might want to remove this column from selected_columns or handle as an error
            
            # Read only the selected columns into a dictionary
            data_dict = {col: all_data_dataset[col][:] for col in selected_columns if col in dataset_fields}
            
            # Create a pandas DataFrame from the dictionary
            data_df = pd.DataFrame(data_dict)
            #data_df = data_df[data_df['station'] == b"KOKB"]  # Filter for 'KOKB' station
            return data_df#[:1_000_000]

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()


def plot_obs_map(data, year, doy):
    """
    Plots the VTEC data on a map using 'lon_ipp', 'lat_ipp' for coordinates.
    """
    sta_data = data.groupby('station').first().reset_index()  # Get the first row for each station

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set map extent (optional, consider if you need a specific region)
    #ax.set_extent([-160, -159, 22, 23], crs=ccrs.PlateCarree()) # Example for global view

    ax.set_global() # Sets the extent to show the whole world

    # --- Basemap Colors (Richer Blue & Beige Pastels) ---
    ocean_color = "#92C6EE" 
    land_color = "#D1B882" 

    ax.add_feature(cfeature.OCEAN, facecolor=ocean_color, edgecolor='none')
    ax.add_feature(cfeature.LAND, facecolor=land_color, edgecolor='none')
    ax.set_facecolor(ocean_color) # Ensure the background of the axes is also ocean color

    # --- Geographic Features (Solid Outlines) ---
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#8A8A8A", zorder=2) # Slightly darker desaturated blue-gray
    
    # --- Gridlines (Solid & Defined) ---
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='#8A8A8A', alpha=1.0, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.x_inline = False
    gl.y_inline = False

    # Set gridline label style (darker text for readability)
    gl.xlabel_style = {'color': "#252525", 'fontsize': 10}
    gl.ylabel_style = {'color': '#252525', 'fontsize': 10}


    # Scatter plot of IPPs with VTEC values
    # Use 'lon_ipp' and 'lat_ipp' from your DataFrame
    sc = ax.scatter(data['lon_ipp'], data['lat_ipp'],s=2,
                    transform=ccrs.PlateCarree(), # Ensure data coordinates are interpreted correctly
                    edgecolor='none', alpha=0.7, label="observation") # Added for better visualization
    
    sc = ax.scatter(sta_data['lon_sta'], sta_data['lat_sta'], s=10, 
                    color='red', marker='o', transform=ccrs.PlateCarree(), label="station")

    # Add color bar
    #plt.colorbar(sc, ax=ax, orientation='vertical', label='VTEC (TECU)')

    # Generate title with date
    date = datetime(year, 1, 1) + pd.Timedelta(days=doy - 1)
    title = f"Observation Map for {date.strftime('%d.%m.%Y')}" # Changed %YY to %Y for full year
    plt.title(title, fontsize=14)

    # Save the figure
    plt.savefig(f"src/data_processing/observation_map_{year}_{doy:03d}.png", bbox_inches='tight', dpi=300)
    plt.close(fig) # Close the figure to free up memory

def plot_obs_density_map_colormap(data, year, doy):
    """
    Plots the density of VTEC observations on a map, with pixel transparency based on density.
    Lower densities will be more transparent, fading to opaque at higher densities,
    using the 'Blues' colormap.
    """
    sta_data = data.groupby('station').first().reset_index()

    fig = plt.figure(figsize=(14, 9)) # Adjusted figure size for better aspect ratio
    ax = plt.axes(projection=ccrs.PlateCarree())

    # --- Map Extent ---
    ax.set_global() # Sets the extent to show the whole world

    # --- Basemap Colors (Richer Blue & Beige Pastels) ---
    ocean_color = "#92C6EE" 
    land_color = "#D1B882" 

    ax.add_feature(cfeature.OCEAN, facecolor=ocean_color, edgecolor='none')
    ax.add_feature(cfeature.LAND, facecolor=land_color, edgecolor='none')
    ax.set_facecolor(ocean_color) # Ensure the background of the axes is also ocean color

    # --- Geographic Features (Solid Outlines) ---
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#8A8A8A", zorder=2) # Slightly darker desaturated blue-gray
    
    # --- Gridlines (Solid & Defined) ---
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='#8A8A8A', alpha=1.0, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.x_inline = False
    gl.y_inline = False

    # Set gridline label style (darker text for readability)
    gl.xlabel_style = {'color': "#252525", 'fontsize': 10}
    gl.ylabel_style = {'color': '#252525', 'fontsize': 10}

    # --- Observation Density Calculation ---
    # For global and fixed bins:
    lon_bins = np.arange(-180, 180.1, 1) # From -180 to 180 with 1-degree step
    lat_bins = np.arange(-90, 90.1, 1)   # From -90 to 90 with 1-degree step

    H, xedges, yedges = np.histogram2d(data['lon_ipp'], data['lat_ipp'],
                                      bins=[lon_bins, lat_bins])

    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2)
    
    # --- Custom Colormap with Alpha Gradient ---
    # 1. Choose base colormap (e.g., 'Blues')
    base_cmap = plt.cm.get_cmap('Blues')
    
    # 2. Define the number of steps/colors in your custom colormap
    n_colors = 256 # Standard resolution for colormaps
    
    # 3. Get the RGB values from the base colormap
    colors = base_cmap(np.linspace(0, 1, n_colors))
    
    # 4. Create an alpha channel that smoothly goes from transparent to opaque
    # from 0.0 (fully transparent) to 1.0 (fully opaque)
    alpha = np.sqrt(np.linspace(0.0, 1.0, n_colors)) ** 0.5

    # 5. Apply the alpha channel to the colors (RGBA)
    colors[:, 3] = alpha
    
    # 6. Create the new custom colormap from these RGBA values
    custom_cmap = mcolors.ListedColormap(colors)

    # --- Set vmin and vmax for the density plot ---
    # vmin=0 means that 0 counts will be fully transparent
    # vmax determines where the color/opacity reaches its maximum. 
    # Using a percentile helps to avoid outliers making the rest of the map too light.
    density_vmin = 0 
    # Set vmax to the 95th percentile of non-zero counts for a good visual range
    density_vmax = np.percentile(H[H > 0], 95) if np.any(H > 0) else 1 
    # Ensure vmax is at least 5 to prevent a very short color scale if data is sparse
    if density_vmax < 5: density_vmax = 5 

    # --- Plot the Observation Density ---
    # Use the custom_cmap with alpha gradient
    im = ax.pcolormesh(X, Y, H.T, cmap=custom_cmap, transform=ccrs.PlateCarree(),
                       zorder=3, # Zorder ensures it's drawn over map features
                       vmin=density_vmin, vmax=density_vmax) 

    # Add a colorbar for the density
    plt.colorbar(im, ax=ax, orientation='vertical', label='Number of Observations per Bin')

    # --- Station Locations ---
    ax.scatter(sta_data['lon_sta'], sta_data['lat_sta'], s=30,
               color='red', marker='o', edgecolor='black', linewidth=0.8,
               transform=ccrs.PlateCarree(), label="GNSS Station", zorder=4) # Highest zorder for visibility

    # Add a legend for stations
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='gray')

    # --- Title ---
    date = datetime(year, 1, 1) + pd.Timedelta(days=doy - 1)
    title = f"Observation Density Map for {date.strftime('%d.%m.%Y')}"
    plt.title(title, fontsize=16, color='#303030', pad=15, fontweight='bold')

    # --- Final Layout and Save ---
    fig.tight_layout(pad=1.5)
    plt.savefig(f"src/data_processing/observation_density_map_{year}_{doy:03d}.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_obs_density_map(data, year, doy):
    """
    Plots the density of VTEC observations on a map, with pixel transparency based on density.
    Lower densities will be more transparent, fading to opaque at higher densities,
    using a single blue color whose transparency is driven by the density.
    """
    sta_data = data.groupby('station').first().reset_index()

    fig = plt.figure(figsize=(14, 9)) # Adjusted figure size for better aspect ratio
    ax = plt.axes(projection=ccrs.PlateCarree())

    # --- Map Extent ---
    ax.set_global() # Sets the extent to show the whole world

    # --- Basemap Colors (Richer Blue & Beige Pastels) ---
    ocean_color = "#92C6EE"
    land_color = "#D1B882"

    ax.add_feature(cfeature.OCEAN, facecolor=ocean_color, edgecolor='none')
    ax.add_feature(cfeature.LAND, facecolor=land_color, edgecolor='none')
    ax.set_facecolor(ocean_color) # Ensure the background of the axes is also ocean color

    # --- Geographic Features (Solid Outlines) ---
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#8A8A8A", zorder=2) # Slightly darker desaturated blue-gray

    # --- Gridlines (Solid & Defined) ---
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='#8A8A8A', alpha=1.0, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.x_inline = False
    gl.y_inline = False

    # Set gridline label style (darker text for readability)
    gl.xlabel_style = {'color': "#252525", 'fontsize': 10}
    gl.ylabel_style = {'color': '#252525', 'fontsize': 10}

    # --- Observation Density Calculation ---
    # For global and fixed bins:
    lon_bins = np.arange(-180, 180.1, 1) # From -180 to 180 with 1-degree step
    lat_bins = np.arange(-90, 90.1, 1)   # From -90 to 90 with 1-degree step

    H, xedges, yedges = np.histogram2d(data['lon_ipp'], data['lat_ipp'],
                                      bins=[lon_bins, lat_bins])

    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2)

    # --- Custom Colormap with Alpha Gradient for a Single Color ---
    # 1. Choose your desired base color in RGB (values from 0 to 1)
    #    This blue aligns well with the "Blues" idea but uses a fixed hue.
    base_rgb_color = mcolors.to_rgb('tab:blue') # Converts 'tab:blue' to its RGB representation
    base_rgb_color = np.array([0.0, 0.0, 0.4])  # Dark blue color in RGB

    # 2. Define the number of steps/colors in your custom colormap
    n_colors = 256  # Standard resolution for colormaps

    # 3. Generate the non-linear alpha channel using our function
    #    Adjust the 'exponent' to control the fall-off (e.g., 2 for moderate, 4 for sharp)
    alpha = np.sqrt(np.linspace(0.0, 1.0, n_colors)) ** 0.5

    # 4. Create an array of RGBA values
    #    Repeat the base RGB color for each step, and apply the corresponding alpha
    colors_rgba = np.zeros((n_colors, 4))
    colors_rgba[:, :3] = base_rgb_color # Set RGB channels for all steps
    colors_rgba[:, 3] = alpha   # Apply the non-linear alpha gradient

    # 5. Create the new custom colormap from these RGBA values
    custom_single_color_cmap = mcolors.ListedColormap(colors_rgba)

    # --- Set vmin and vmax for the density plot ---
    # vmin=0 means that 0 counts will be fully transparent
    # vmax determines where the color/opacity reaches its maximum.
    # Using a percentile helps to avoid outliers making the rest of the map too light.
    density_vmin = 0
    # Set vmax to the 95th percentile of non-zero counts for a good visual range
    density_vmax = np.percentile(H[H > 0], 95) if np.any(H > 0) else 1
    # Ensure vmax is at least 5 to prevent a very short color scale if data is sparse
    if density_vmax < 5: density_vmax = 5

    # --- Plot the Observation Density ---
    # Use the custom_single_color_cmap with fixed color and alpha gradient
    im = ax.pcolormesh(X, Y, H.T, cmap=custom_single_color_cmap, transform=ccrs.PlateCarree(),
                       zorder=3, # Zorder ensures it's drawn over map features
                       vmin=density_vmin, vmax=density_vmax)

    # Add a colorbar for the density
    plt.colorbar(im, ax=ax, orientation='vertical', label='Number of Observations per Bin (Transparency Indicates Density)')

    # --- Station Locations ---
    ax.scatter(sta_data['lon_sta'], sta_data['lat_sta'], s=30,
               color='red', marker='o', edgecolor='black', linewidth=0.8,
               transform=ccrs.PlateCarree(), label="GNSS Station", zorder=4) # Highest zorder for visibility

    # Add a legend for stations
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='gray')

    # --- Title ---
    date = datetime(year, 1, 1) + pd.Timedelta(days=doy - 1)
    title = f"Observation Density Map for {date.strftime('%d.%m.%Y')}"
    plt.title(title, fontsize=16, color='#303030', pad=15, fontweight='bold')

    # --- Final Layout and Save ---
    fig.tight_layout(pad=1.5)
    plt.savefig(f"src/data_processing/observation_density_map_{year}_{doy:03d}.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

def main():
    # File path to the VTEC data file
    data_folder = "/home/space/data/IONO/STEC_DB"

    # Desired date (year and day of year)
    year = 2023
    doy = 150

    # Read the data
    data = read_vtec_data(data_folder, year, doy)
    if data is None:
        return

    # Plot the data
    plot_obs_density_map(data, year, doy)
    plot_obs_map(data, year, doy)
    
    print(f"Plot saved for {year}-{doy:03d}.")

if __name__ == "__main__":
    main()