import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import netCDF4 as nc
import os
import configparser
from spacepy.coordinates import Coords
from spacepy.time import Ticktock
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


config = configparser.ConfigParser()
config.read('settings.ini')
main_dir=config['Main']['main_path']
data_dir=config['Main']['data_path']
sa_code_dir = os.path.join(main_dir,'sa_code')
sa_data_dir = os.path.join(data_dir,'jason3')



def convert_to_datetime(row):
    date = pd.Timestamp(row['y'], 1, 1) + pd.to_timedelta(row['doy'] - 1, unit='D')
    time = pd.to_timedelta(row['sod'], unit='s')
    return date + time

def coord_tranform(inp_type, out_type, lats, lons, epochs):
    n = lats.shape[0]
    coords = np.zeros((n, 3))
    for i in range(n):
        coords[i] = [1+450/6371, lats[i], lons[i]]
    coords = list(coords)
    coord_inp = Coords(coords, inp_type, 'sph')
    coord_inp.ticks = Ticktock(epochs, 'UTC')
    
    coord_out = coord_inp.convert(out_type, 'sph')
    return coord_out

def correct_longitude_sa(lon):
    if lon > 180:
        return lon -360
    else:
        return lon



fku = 13.575
fcs = 5.41 # Sentinel-3
k = 0.40250
scale = -fku*fku/k

columns = ['lon','lat','time','dion','vtec']
df_total = pd.DataFrame(columns=columns)

cwd = os.getcwd()
for cycle in os.listdir(sa_data_dir):
    print(cycle)
    cycle_dir = os.path.join(sa_data_dir,cycle)
    if os.path.isdir(cycle_dir):
        for ncfile in os.listdir(cycle_dir):
            if ncfile.endswith('.nc'):
                ncfile_dir = os.path.join(cycle_dir,ncfile)
                ds = nc.Dataset(ncfile_dir)
                lon = ds['lon'][:]
                lat = ds['lat'][:]
                time = ds['time'][:]
                iono_alt_smooth = ds['iono_alt_smooth'][:]
                vtec = iono_alt_smooth*scale


                unix_epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
                start_epoch = datetime(1985, 1, 1, tzinfo=timezone.utc)
                delta = start_epoch.timestamp() - unix_epoch.timestamp()
                time = time + delta
                df = {'lon': lon, 'lat': lat, 'time': time, 'dion': iono_alt_smooth,'vtec':vtec}
                df = pd.DataFrame(df)
                df = df.dropna(how='any')
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df['lon'] = df['lon'].apply(lambda x: correct_longitude_sa(x))

                lats = df['lat'].values.astype(np.float32)
                lons = df['lon'].values.astype(np.float32)
                epochs = list(df['time'].values.astype('str'))
                inp_type = "GEO"
                out_type = "SM"
                coord_sm = coord_tranform(inp_type, out_type, lats, lons, epochs)
                df['sm_lat'] = coord_sm.lati
                df['sm_lon'] = coord_sm.long


                df_total = df_total._append(df,ignore_index=True)


#df_total = df_total.dropna(how='any')
#df_total['time'] = pd.to_datetime(df_total['time'], unit='s')
df_total = df_total.sort_values(by='time')


"""lats = df_total['lat'].values.astype(np.float32)
lons = df_total['lon'].values.astype(np.float32)
epochs = list(df_total['time'].values.astype('str'))
inp_type = "GEO"
out_type = "SM"
coord_sm = coord_tranform(inp_type, out_type, lats, lons, epochs)
df_total['sm_lat'] = coord_sm.lati
df_total['sm_lon'] = coord_sm.long"""



#df_total.to_csv('test_all.csv', index=None, float_format='%.5f')
df_total.to_csv(os.path.join(sa_data_dir,'sa_dataset.csv'),index=False)