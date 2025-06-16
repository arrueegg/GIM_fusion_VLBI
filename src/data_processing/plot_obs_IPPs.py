import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import os
import re
import json
import tarfile
import netCDF4 as nc
from datetime import datetime, timedelta
from io import StringIO
from spacepy.coordinates import Coords
from spacepy.time import Ticktock
import pyproj

def read_gnss_data(folder, year, doy):
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


def read_vlbi_vtec_data(data_folder, year, doy):
    """
    Read VLBI VTEC time-series data into a pandas DataFrame.

    Parameters:
        data_folder (str): Root folder containing 'SX' and 'VGOS' subfolders and a 'station_coords.json' file.
        year (int or str): Four-digit year of observation.
        doy (int or str): Day-of-year (1-366) of observation.

    Returns:
        pd.DataFrame
    """
    year = str(year)
    doy = int(doy)
    # Prepare dates
    date1 = datetime(int(year), 1, 1) + timedelta(days=doy-1)
    date2 = date1 - timedelta(days=1)
    name1 = date1.strftime('%Y%m%d')
    name2 = date2.strftime('%Y%m%d')

    # Find summary.md paths
    summary_paths = []
    for mode in ('SX', 'VGOS'):
        root = os.path.join(data_folder, mode, year)
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            if entry.startswith(name1) or entry.startswith(name2):
                summary = os.path.join(root, entry, 'summary.md')
                if os.path.isfile(summary):
                    summary_paths.append(summary)

    # Load station coordinates
    coords_file = os.path.join(data_folder, 'station_coords.json')
    with open(coords_file, 'r') as f:
        sta_coords = json.load(f)

    # Extract tables
    dfs = []
    table_re = re.compile(
        r'^##\s*VTEC Time Series\s*\n(?P<table>[\s\S]*?)(?=^##\s|\Z)',
        re.MULTILINE
    )
    for path in summary_paths:
        text = open(path, 'r', encoding='utf-8').read()
        match = table_re.search(text)
        if not match:
            continue
        lines = [ln for ln in match.group('table').splitlines() if ln.strip()]
        md = "\n".join(lines)
        df = pd.read_csv(
            StringIO(md), sep='|', header=0, skiprows=[1], engine='python'
        )
        df = df.iloc[:, 1:-1]
        df.columns = df.columns.str.strip()
        df = df.applymap(lambda v: v.strip() if isinstance(v, str) else v)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)

    df.rename(columns={'vlbi_vtec': 'vtec',}, inplace=True)

    # Add station geodetic coords
    df['lat'] = df['station'].map(lambda x: sta_coords.get(x, {}).get('Latitude'))
    df['lon'] = df['station'].map(lambda x: sta_coords.get(x, {}).get('Longitude'))

    # Parse date and epoch
    df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
    df['epoch'] = pd.to_datetime(
        df['date'].dt.strftime('%Y-%m-%d') + ' ' + df['epoch'].astype(str),
        format='%Y-%m-%d %H:%M:%S'
    )
    df['doy'] = df['date'].dt.dayofyear
    df = df[df['doy'] == doy]

    # GEO -> SM conversion
    coords = np.column_stack((
        np.full(len(df), 1 + 450 / 6371),
        df['lat'], df['lon']
    ))
    sm = Coords(coords, 'GEO', 'sph')
    sm.ticks = Ticktock(
        df['epoch'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(), 'UTC'
    )
    sm_out = sm.convert('SM', 'sph')
    df['sm_lat'] = sm_out.lati.astype(float)
    df['sm_lon'] = ((sm_out.long.astype(float) + 180) % 360) - 180

    # Temporal features
    df['sod'] = (
        df['epoch'].dt.hour * 3600 + df['epoch'].dt.minute * 60 + df['epoch'].dt.second
    )
    
    return df[['station', 'date', 'epoch', 'vtec', 'sm_lat', 'sm_lon',
               'sod', 'lat', 'lon']]

def read_vlbi_dtec_data(data_folder, year, doy):
    """
    Read VLBI DTEC data into a pandas DataFrame, separately handling SX & VGOS sessions.

    Parameters:
        data_folder (str): Root folder with subfolders 'SX' and 'VGOS', each with a '<year>' folder of tgz files.
        year (int or str): Year of observation.
        doy (int or str): Day-of-year of observation.
    Returns:
        pd.DataFrame
    """
    year = str(year)
    doy = int(doy)
    date1 = datetime(int(year),1,1) + timedelta(days=doy-1)
    date2 = date1 - timedelta(days=1)
    prefixes = {date1.strftime('%Y%m%d'), date2.strftime('%Y%m%d')}

    temp = os.path.join(data_folder, 'temp_extracted')
    os.makedirs(temp, exist_ok=True)

    sessions = []
    for mode in ('SX','VGOS'):
        root = os.path.join(data_folder, year)
        for fname in os.listdir(root):
            if any(fname.startswith(p) for p in prefixes) and fname.endswith('.tgz'):
                with tarfile.open(os.path.join(root,fname),'r:gz') as tar:
                    tar.extractall(temp)
                sessions.append((mode, os.path.join(temp, os.path.splitext(fname)[0])))

    all_dfs = []
    for mode, session in sessions:
        # Read dTEC
        try:
            if mode == 'VGOS':
                ds = nc.Dataset(os.path.join(session, 'Observables', 'DiffTec.nc'))
                dtec = -ds['diffTec'][:].data
                ds.close()
            else:
                ds = nc.Dataset(os.path.join(session, 'ObsDerived', 'Cal-SlantPathIonoGroup_bX.nc'))
                freq = nc.Dataset(os.path.join(session, 'Observables', 'RefFreq_bX.nc'))['RefFreq'][:].data * 1e6
                raw = ds['Cal-SlantPathIonoGroup'][:, 0].data
                dtec = raw * 299792458 * freq**2 * 1e-16 / 40.31
                ds.close()
        except:
            continue
        # Read epochs
        t = nc.Dataset(os.path.join(session, 'Observables', 'TimeUTC.nc'))
        YMDHM = t['YMDHM'][:].data
        secs = t['Second'][:].data
        t.close()
        epochs = [
            datetime(int(r[0]) + 2000 if int(r[0]) < 1000 else int(r[0]),
                    int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(s))
            for r, s in zip(YMDHM, secs)
        ]
        sod = np.array([e.hour * 3600 + e.minute * 60 + e.second for e in epochs])
        # Cross-references
        try:
            ox = nc.Dataset(os.path.join(session, 'CrossReference', 'ObsCrossRef.nc'))
            Obs2Scan = ox['Obs2Scan'][:].data
            Obs2Base = ox['Obs2Baseline'][:].data
            if Obs2Base.ndim == 1:
                Obs2Base = Obs2Base.reshape(-1, 2)
                Obs2Base = np.repeat(Obs2Base, Obs2Scan.shape[0], axis=0)
            ox.close()
        except:
            continue
        stcr = nc.Dataset(os.path.join(session, 'CrossReference', 'StationCrossRef.nc'))
        raw_sta = stcr['CrossRefStationList'][:]
        Scan2Sta = stcr['Scan2Station'][:].data
        stcr.close()
        stations = [''.join(c.decode('utf-8').strip() for c in rec) for rec in raw_sta]
        # Station geodetic coords
        try:
            apr = nc.Dataset(os.path.join(session, 'Apriori', 'Station.nc'))
            stationXYZ = apr['AprioriStationXYZ'][:]
        except:
            continue
        transformer = pyproj.Transformer.from_crs({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
                                                {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'})
        # latitude, longitude, and altitude order
        stationLLA = np.zeros(shape=(len(stationXYZ[:]), len(stationXYZ[0, :])))
        # cartesian(XYZ) to georgraphic(LLA) coordinates
        for n, item in enumerate(stationXYZ[:]):
            # longitude, latitude, and altitude order
            stationLLA[n, 1], stationLLA[n, 0], stationLLA[n, 2] = transformer.transform(item[0], item[1], item[2])
            
        # Az/El
        AzEl = np.zeros((len(Obs2Base), 4))
        for i, (s1, s2) in enumerate(Obs2Base):
            for idx, (sta, col) in enumerate(((s1, 0), (s2, 2))):
                sd = stations[sta-1].upper()
                azf = nc.Dataset(os.path.join(session, sd, 'AzEl.nc'))
                AzTheo = azf['AzTheo'][:].data
                ElTheo = azf['ElTheo'][:].data
                azf.close()
                scan_i = Scan2Sta[Obs2Scan[i]-1, sta-1] - 1
                AzEl[i, col] = AzTheo[scan_i][0]
                AzEl[i, col+1] = ElTheo[scan_i][0]

        # Assemble DataFrame
        df = pd.DataFrame({
            'dtec': dtec,
            'epoch': epochs,
            'sod': sod,
            'sta1': [stations[b[0]-1] for b in Obs2Base],
            'sta2': [stations[b[1]-1] for b in Obs2Base],
            'sta1_lat': [stationLLA[b[0]-1, 0] for b in Obs2Base],
            'sta1_lon': [stationLLA[b[0]-1, 1] for b in Obs2Base],
            'sta2_lat': [stationLLA[b[1]-1, 0] for b in Obs2Base],
            'sta2_lon': [stationLLA[b[1]-1, 1] for b in Obs2Base],
            'Az_sta1': AzEl[:, 0],
            'El_sta1': AzEl[:, 1],
            'Az_sta2': AzEl[:, 2],
            'El_sta2': AzEl[:, 3],
        })
        # Filter by DOY
        df['epoch'] = pd.to_datetime(df['epoch'])
        df = df[df['epoch'].dt.dayofyear == doy]
        if df.empty:
            continue
        # Compute geographic IPP coordinates only
        R, h = 6371.0, 450.0
        for label in ('1', '2'):
            El = df[f'El_sta{label}']
            Az = df[f'Az_sta{label}']
            Psi = np.pi/2 - El - np.arcsin(R/(R+h) * np.cos(El))
            lat0 = np.deg2rad(df[f'sta{label}_lat'])
            lon0 = np.deg2rad(df[f'sta{label}_lon'])
            Phi = np.arcsin(np.sin(lat0)*np.cos(Psi) + np.cos(lat0)*np.sin(Psi)*np.cos(Az))
            Lam = lon0 + np.arcsin(np.sin(Psi)*np.sin(Az)/np.cos(Phi))
            df[f'IPP{label}_lat'] = np.rad2deg(Phi)
            df[f'IPP{label}_lon'] = np.rad2deg(Lam)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

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


def plot_obs_density_map(gnss_data, vlbi_vtec_data, vlbi_dtec_data, year, doy, plot_vtec=True, plot_dtec=True):

    """
    Plots the density of VTEC observations on a map, with pixel transparency based on density.
    Lower densities will be more transparent, fading to opaque at higher densities,
    using a single blue color whose transparency is driven by the density.
    """
    sta_data = gnss_data.groupby('station').first().reset_index()
    if plot_vtec:
        vlbi_sta_data = vlbi_vtec_data.groupby('station').first().reset_index()
    if plot_dtec:
        vlbi_dtec_sta1 = (
            vlbi_dtec_data[['sta1','sta1_lat','sta1_lon']]
            .rename(columns={
                'sta1': 'station',
                'sta1_lat': 'lat',
                'sta1_lon': 'lon'
            })
        )

        vlbi_dtec_sta2 = (
            vlbi_dtec_data[['sta2','sta2_lat','sta2_lon']]
            .rename(columns={
                'sta2': 'station',
                'sta2_lat': 'lat',
                'sta2_lon': 'lon'
            })
        )

        stations_dtec = (
            pd.concat([vlbi_dtec_sta1, vlbi_dtec_sta2], ignore_index=True)
            .drop_duplicates(subset=['station'])
            .reset_index(drop=True)
        )
    if plot_vtec and plot_dtec:
        vlbi_sta_data = pd.concat([vlbi_sta_data, stations_dtec], ignore_index=True).drop_duplicates(subset=['station']).reset_index(drop=True)

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
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#8A8A8A", zorder=4) # Slightly darker desaturated blue-gray

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

    H, xedges, yedges = np.histogram2d(gnss_data['lon_ipp'], gnss_data['lat_ipp'],
                                      bins=[lon_bins, lat_bins])

    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2)

    # --- Custom Colormap with Alpha Gradient for a Single Color ---
    # 1. Choose your desired base color in RGB (values from 0 to 1)
    #    This blue aligns well with the "Blues" idea but uses a fixed hue.
    base_rgb_color = mcolors.to_rgb('tab:blue') # Converts 'tab:blue' to its RGB representation
    base_rgb_color = np.array([0.0, 0.2, 1.0])  # Dark blue color in RGB

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
    plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.6, label='Number of GNSS Observations per 1°x1° Bin')

    # --- Station Locations ---
    ax.scatter(sta_data['lon_sta'], sta_data['lat_sta'], s=30,
               color='red', marker='o', edgecolor='face', linewidth=0.8,
               transform=ccrs.PlateCarree(), label="GNSS Station", zorder=5) # Highest zorder for visibility
    if plot_vtec or plot_dtec:

        if plot_vtec:
            ax.scatter(vlbi_sta_data['lon'], vlbi_sta_data['lat'], s=50,
                        color='yellow', marker='*', edgecolor='face', linewidth=0.8,
                        transform=ccrs.PlateCarree(), label="VLBI Station", zorder=6) # Highest zorder for visibility

        if plot_dtec and not plot_vtec:
            ax.scatter(stations_dtec['lon'], stations_dtec['lat'], s=50,
                        color='orange', marker='*', edgecolor='face', linewidth=0.8,
                        transform=ccrs.PlateCarree(), label="VLBI DTEC Station", zorder=6)
 
        from pyproj import Geod
        geod = Geod(ellps='WGS84')

        # for each VLBI station:
        count=0
        for lon0, lat0 in zip(vlbi_sta_data['lon'], vlbi_sta_data['lat']):
            # sample bearings from 0° to 360° in fine steps
            bearings = np.linspace(0, 360, 361)
            # compute the circle: returns lon, lat (and back‐az, unused)
            lons_circ, lats_circ, _ = geod.fwd(
                np.full_like(bearings, lon0),
                np.full_like(bearings, lat0),
                bearings,
                np.full_like(bearings, 1_000_000)  # distance in meters
            )

            labelboarder = f"1'000km radius around VLBI station" if count==0 else None
            count+=1
            # plot the circle
            ax.plot(lons_circ, lats_circ,
                    transform=ccrs.PlateCarree(), color='yellow',   
                    linewidth=1.5, zorder=6, label=labelboarder)  
        
    if plot_dtec:
        # Plot VLBI DTEC observations
        ax.scatter(vlbi_dtec_data['IPP1_lon'], vlbi_dtec_data['IPP1_lat'], s=5,
                   color='tab:orange', marker='o', edgecolor='face', linewidth=0.8,
                   transform=ccrs.PlateCarree(), label="VLBI DTEC Observations", zorder=4)
        ax.scatter(vlbi_dtec_data['IPP2_lon'], vlbi_dtec_data['IPP2_lat'], s=5,
                   color='tab:orange', marker='o', edgecolor='face', linewidth=0.8,
                   transform=ccrs.PlateCarree(), zorder=4)

    # Add a legend for stations
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='gray')

    # --- Save without title ---
    fig.tight_layout(pad=1.5)
    plt.savefig(f"src/data_processing/observation_density_map_{year}_{doy:03d}_notitle.png", bbox_inches='tight', dpi=300)

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
    gnss_data_folder = "/home/space/data/IONO/STEC_DB"
    vlbi_vtec_data_path = "/scratch2/arrueegg/WP1/VLBIono/Results/"
    vlbi_dtec_data_path = "/home/space/data/vlbi/ivsdata/vgosdb/"

    # Desired date (year and day of year)
    year = 2023
    doy = 180

    # Read the data
    vlbi_dtec_data = read_vlbi_dtec_data(vlbi_dtec_data_path, year, doy)
    vlbi_vtec_data = read_vlbi_vtec_data(vlbi_vtec_data_path, year, doy)
    gnss_data = read_gnss_data(gnss_data_folder, year, doy)
    if gnss_data is None:
        return

    # Plot the data
    plot_obs_density_map(gnss_data, vlbi_vtec_data, vlbi_dtec_data, year, doy, plot_vtec=True, plot_dtec=True)
    plot_obs_map(gnss_data, year, doy)
    
    print(f"Plot saved for {year}-{doy:03d}.")

if __name__ == "__main__":
    main()