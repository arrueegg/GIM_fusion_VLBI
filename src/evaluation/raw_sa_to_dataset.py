import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import netCDF4 as nc
from spacepy.coordinates import Coords
from spacepy.time import Ticktock
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Directories
MAIN_DIR = "/scratch2/arrueegg/WP2/GIM_Fusion_VLBI/"
DATA_DIR = "/home/ggl/project/miten/jason3_data/"
SA_DATA_DIR = os.path.join(DATA_DIR, 'jason3')

# Constants
FKU = 13.575
FCS = 5.41  # Sentinel-3
K = 0.40250
SCALE = -FKU**2 / K

# Helper Functions
def convert_to_datetime(row):
    return pd.Timestamp(row['y'], 1, 1) + pd.to_timedelta(row['doy'] - 1, unit='D') + pd.to_timedelta(row['sod'], unit='s')

def coord_transform(inp_type, out_type, lats, lons, epochs):
    coords = np.column_stack((np.ones_like(lats) + 450 / 6371, lats, lons))
    coord_inp = Coords(coords.tolist(), inp_type, 'sph')
    coord_inp.ticks = Ticktock(epochs, 'UTC')
    coord_out = coord_inp.convert(out_type, 'sph')
    return coord_out

def process_nc_file(ncfile_path):
    ds = nc.Dataset(ncfile_path)
    lon, lat, time, iono_alt_smooth = ds['lon'][:], ds['lat'][:], ds['time'][:], ds['iono_alt_smooth'][:]
    vtec = iono_alt_smooth * SCALE

    unix_epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    start_epoch = datetime(1985, 1, 1, tzinfo=timezone.utc)
    delta = (start_epoch - unix_epoch).total_seconds()
    time += delta

    df = pd.DataFrame({
        'lon': lon,
        'lat': lat,
        'time': pd.to_datetime(time, unit='s'),
        'dion': iono_alt_smooth,
        'vtec': vtec
    }).dropna()

    df['lon'] = df['lon'] % 360
    return df

# Main Processing
def main():
    columns = ['lon', 'lat', 'time', 'dion', 'vtec', 'sm_lat', 'sm_lon']
    df_total = pd.DataFrame(columns=columns)

    for cycle in os.listdir(SA_DATA_DIR):
        cycle_dir = os.path.join(SA_DATA_DIR, cycle)
        if os.path.isdir(cycle_dir):
            for ncfile in os.listdir(cycle_dir):
                if ncfile.endswith('.nc'):
                    ncfile_path = os.path.join(cycle_dir, ncfile)
                    df = process_nc_file(ncfile_path)

                    # Coordinate Transformation
                    lats, lons = df['lat'].values.astype(np.float32), df['lon'].values.astype(np.float32)
                    epochs = df['time'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
                    coord_sm = coord_transform("GEO", "SM", lats, lons, epochs)

                    df['sm_lat'], df['sm_lon'] = coord_sm.lati, coord_sm.long
                    df_total = pd.concat([df_total, df], ignore_index=True)

    df_total.sort_values(by='time', inplace=True)
    output_path = os.path.join(SA_DATA_DIR, 'sa_dataset.csv')
    df_total.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
