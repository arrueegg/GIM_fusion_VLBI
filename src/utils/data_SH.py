import os
import stat
import h5py
from matplotlib.pylab import f
import pandas as pd
from requests import get
import torch
import numpy as np
import tarfile
import netCDF4 as nc
import pyproj
import re
from io import StringIO
from datetime import datetime, timedelta
from spacepy.coordinates import Coords
from spacepy.time import Ticktock
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from utils.locationencoder.pe import SphericalHarmonics

import warnings
warnings.filterwarnings("ignore")

class SingleGNSSDataset(Dataset):
    def __init__(self, config, split='random'):
        # Load and preprocess your data here
        self.year = config['year']
        self.doy = config['doy']
        self.elev = config['preprocessing']['elevation']  # Elevation threshold
        self.columns_to_load = config["data"]["columns_to_load"]
        
        data_file = os.path.join(config['data']['GNSS_data_path'], str(self.year), str(self.doy), f'ccl_{self.year}{self.doy}_30_5.h5')
        self.split = split
        self.mode = config['data']['mode']
        self.data = self.load_data(data_file)
    
    def __len__(self):
        return len(self.data)

    def load_data(self, data_file):

        if self.split != 'random':
            sta_list = np.loadtxt(f'./src/data_processing/{self.split}.list', dtype=str)
            
        data = {}

        with h5py.File(data_file, 'r') as h5_file:
            
            all_data = h5_file[self.year][self.doy]['all_data']

            # If no specific columns are provided, load all columns
            if self.columns_to_load is None:
                self.columns_to_load = list(all_data.dtype.names)
            
            if self.split != 'random':
                # Load the 'station' column to filter rows first
                station_column = all_data['station'][:]
                station_column = np.array([x.decode('utf-8').upper() if isinstance(x, bytes) else x for x in station_column])
        
                # Filter indices for the wanted stations
                wanted_indices = np.isin(station_column, sta_list)

                # Load only the filtered rows for each column
                for column in self.columns_to_load:
                    data[column] = all_data[column][wanted_indices]

            else:
                # Load the data for the specified columns
                for column in self.columns_to_load:
                    data[column] = all_data[column][:]

        
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(data)

        df = self.filter_df(df)

        return df
    
    def filter_df(self, df):
        
        df['station'] = df['station'].apply(lambda x: x.decode('utf-8').upper() if isinstance(x, bytes) else x)

        if self.split != 'random':
            sta_list = np.loadtxt(f'./src/data_processing/{self.split}.list', dtype=str)
            df = df[df['station'].isin(sta_list)]

        # Filter data
        mask = (abs(df['dcbs']) > 1e-3) & (abs(df['dcbr']) > 1e-3) & (df['vtec'] > 2.0) & \
            (df['vtec'] <= 200) & (df['satele'] >= self.elev)
        df = df[mask]
        
        # Handle empty dataframe case
        if df.empty:
            raise ValueError("DataFrame is empty after filtering, check your filtering conditions or data.")

        df.loc[:, 'sin_utc'] = np.sin(df['sod'] / 86400 * 2 * np.pi)
        df.loc[:, 'cos_utc'] = np.cos(df['sod'] / 86400 * 2 * np.pi)
        df.loc[:, 'sod_normalize'] = 2 * df['sod'] / 86400 - 1

        # Normalize spatial features
        df.loc[:, 'sm_lon_ipp'] = (df['sm_lon_ipp'] + 180) % 360 - 180

        # Keep only the necessary columns
        columns_to_keep = ['vtec', 'sm_lat_ipp', 'sm_lon_ipp', 'sin_utc', 'cos_utc', 'sod_normalize']
        if self.mode == 'DTEC_Fusion':
            columns_to_keep.append('satele')
            columns_to_keep.append('station')
            df['station'] = -1
        return torch.tensor(df[columns_to_keep].values, dtype=torch.float32)


    def __getitem__(self, idx):

        x = self.data[idx, 1:]  # All columns except the first one as features
        y = self.data[idx, 0]   # First column as the label

        tech = torch.tensor(0, dtype=torch.int64)  # 0 == GNSS

        return x, y, tech
    
class SingleVLBIDataset(Dataset):
    def __init__(self, config, split='random'):
        # Load and preprocess your data here
        self.year = config['year']
        self.doy = config['doy']
        self.vlbi_path = config['data']['VLBI_data_path']

        data_files = self.get_file_path(config)
        self.split = split
        self.data = self.load_data(data_files)
    
    def __len__(self):
        return len(self.data)

    def get_file_path(self, config):
        
        date1 = datetime(int(self.year), 1, 1) + timedelta(days=int(self.doy) - 1)
        date2 = date1 - timedelta(days=1)
        name1 = date1.strftime('%Y%m%d').upper()
        name2 = date2.strftime('%Y%m%d').upper()
        vlbi_path = os.path.join(config["data"]["VLBI_data_path"], "SX", f"{self.year}")
        vgos_path = os.path.join(config["data"]["VLBI_data_path"], "VGOS", f"{self.year}")

        paths = []
        for _, dirs, _ in os.walk(vlbi_path):
            for dir_name in dirs:
                if dir_name.startswith(name1) or dir_name.startswith(name2): 
                    summary_path = os.path.join(vlbi_path, dir_name, 'summary.md')
                    if os.path.exists(summary_path):
                        paths.append(summary_path)  
        
        for _, dirs, _ in os.walk(vgos_path):
            for dir_name in dirs:
                if dir_name.startswith(name1) or dir_name.startswith(name2):  
                    summary_path = os.path.join(vgos_path, dir_name, 'summary.md')
                    if os.path.exists(summary_path):
                        paths.append(summary_path)
        return paths

    def load_data(self, data_files):
        data = pd.DataFrame()
        if len(data_files) == 0:
            return data
        data_list = [self.load_markdown(file) for file in data_files]
        data = pd.concat(data_list, ignore_index=True)

        data = self.preprocess(data)
        return data

    def load_markdown(self, path):
        """
        Load a markdown file of session results and process its contents into separate DataFrames.

        Parameters:
            path (str): The path of the markdown file.

        Returns:
            'df_vtecs' (DataFrame): VTEC time-series table.

        """
        # Read entire file
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # find section and capture everything until the next "## " or end-of-file
        pattern = re.compile(
            rf'^##\s*{re.escape("VTEC Time Series")}\s*\n'      # start of the named section
            r'(?P<table>[\s\S]*?)(?=^##\s|\Z)',             # everything up to next "## " or EOF
            re.MULTILINE
        )
        m = pattern.search(content)
        if not m:
            return pd.DataFrame()  # empty if not found

        # get just the markdown table text, line-by-line
        raw = m.group('table').strip().splitlines()
        # drop any blank lines
        lines = [ln for ln in raw if ln.strip()]
        md_table = "\n".join(lines)

        # read with pandas (separator is '|' in markdown tables)
        df = pd.read_csv(
            StringIO(md_table),
            sep='|',
            header=0,
            skiprows=[1],        # skip the markdown separator row (----|----)
            engine='python'
        )
        # drop the empty first/last columns created by leading/trailing pipes
        df = df.iloc[:, 1:-1]
        # strip whitespace from column names & cell values
        df.columns = df.columns.str.strip()
        df = df.applymap(lambda v: v.strip() if isinstance(v, str) else v)
        # convert any column that can be numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

        return df

    def preprocess(self, df):
        if self.split != 'random':
            file_path = f'./src/data_processing/sit_{self.split}_vlbi.list'
            if os.path.getsize(file_path) > 0:  # Check if the file is not empty
                sta_list = np.atleast_1d(np.loadtxt(file_path, dtype=str))
            else:
                sta_list = np.array([]) 
            df = df[df['station'].isin(sta_list)].copy()

        # Handle empty dataframe case
        if df.empty:
            return df

        # Filter data
        df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
        df.loc[:, 'doy'] = df['date'].dt.dayofyear
        mask = df['doy'] == int(self.doy) 
        df = df[mask]

        sta_coords = pd.read_json(os.path.join("src", "utils", "station_coords.json"))
        
        df['Latitude'] = df['station'].map(lambda x: sta_coords.get(x, {}).get('Latitude'))
        df['Longitude'] = df['station'].map(lambda x: sta_coords.get(x, {}).get('Longitude'))

        lats = df['Latitude'].values
        lons = df['Longitude'].values
        coords = np.column_stack((np.full_like(lats, 1 + 450 / 6371), lats, lons))
        # Convert 'date' to string and 'epoch' to a datetime object
        epochs = list(pd.to_datetime(df['date'].astype(str) + ' ' + df['epoch'].astype(str)).dt.strftime('%Y-%m-%d %H:%M:%S'))
        
        coord_sm = self.coord_transform(coords, epochs, "GEO", "SM")  
        sm_lats = coord_sm.lati.astype(np.float32)
        sm_lons = coord_sm.long.astype(np.float32)

        df['sm_lat'] = np.clip(sm_lats, -90, 90)
        df['sm_lon'] = ((sm_lons + 180) % 360) - 180

        times = pd.to_datetime(df['epoch'], format='%H:%M:%S')
        df['sod'] = times.dt.hour * 3600 + times.dt.minute * 60 + times.dt.second

        df.rename(columns={'vgos_vtec': 'vtec'}, inplace=True)
        df.rename(columns={'vlbi_vtec': 'vtec'}, inplace=True)

        # Filter data
        mask = (df['vtec'] > 0.1) & (df['vtec'] <= 200)
        df = df[mask]
        
        # Handle empty dataframe case
        if df.empty:
            return df

        # Normalize temporal features
        df.loc[:, 'sin_utc'] = np.sin(df['sod'] / 86400 * 2 * np.pi)
        df.loc[:, 'cos_utc'] = np.cos(df['sod'] / 86400 * 2 * np.pi)
        df.loc[:, 'sod_normalize'] = 2 * df['sod'] / 86400 - 1

        # Keep only the necessary columns
        columns_to_keep = ['vtec', 'sm_lat', 'sm_lon', 'sin_utc', 'cos_utc', 'sod_normalize']
        return torch.tensor(df[columns_to_keep].values, dtype=torch.float32)
        
    def coord_transform(self, coords, epochs, inp_type, out_type):
        coord_inp = Coords(coords, inp_type, 'sph')
        coord_inp.ticks = Ticktock(epochs, 'UTC')
        return coord_inp.convert(out_type, 'sph')

    def __getitem__(self, idx):

        x = self.data[idx, 1:]  # All columns except the first one as features
        y = self.data[idx, 0]   # First column as the label

        tech = torch.tensor(1, dtype=torch.int64)  # 1 == VLBI
        return x, y, tech
    
class DTECVLBIDataset(Dataset):
    def __init__(self, config, split='random'):
        # Load and preprocess your data here
        self.year = config['year']
        self.doy = config['doy']
        self.vlbi_path = config['data']['VLBI_raw_data_path']
        self.vlbi_types = ['r1', 'r4', 'vo']

        data_paths = self.get_file_path(config)
        self.split = split
        self.data = self.load_data(data_paths)
        self.stations = self.get_station_list(data_paths)

    def __len__(self):
        return len(self.data)
    
    def get_file_path(self, config):
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)

        date1 = datetime(int(self.year), 1, 1) + timedelta(days=int(self.doy) - 1)
        date2 = date1 - timedelta(days=1)
        date_str1 = date1.strftime('%Y%m%d')
        date_str2 = date2.strftime('%Y%m%d')
        vlbi_path = os.path.join(config["data"]["VLBI_raw_data_path"], f"{self.year}")

        paths = []
        for zipfile in os.listdir(vlbi_path):
            if zipfile.startswith(date_str1) or zipfile.startswith(date_str2):
                for vlbi_type in self.vlbi_types:
                    if vlbi_type in zipfile:
                        zip_path = os.path.join(vlbi_path, zipfile)
                        with tarfile.open(zip_path, 'r:gz') as tar:
                            tar.extractall(path=temp_dir)
                        paths.append(os.path.join(temp_dir, zipfile.split('.')[0]))
        return paths
    
    def read_dTEC(self, path):
        '''This function reads the ionospheric delays (dTEC) of VGOS or geodetic VLBI observations.'''

        freq = self.get_freq(path)
        tech = 'VGOS' if 'vo' in path else 'VLBI' if 'r1' in path or 'r4' in path else None

        try:
            if tech == 'VGOS':
                ds = nc.Dataset(os.path.join(path, 'Observables', 'DiffTec.nc'))
                dtec = ds['diffTec'][:].data
                dtec = -dtec
                dtecstd = ds['diffTecStdDev'][:].data
            elif tech == 'VLBI':
                ds = nc.Dataset(os.path.join(path, 'ObsDerived', 'Cal-SlantPathIonoGroup_bX.nc'))
                dtec = ds['Cal-SlantPathIonoGroup'][:, 0].data
                dtecstd = ds['Cal-SlantPathIonoGroupSigma'][:, 0].data
                dtec = dtec * 299792458 * freq**2 * (10**-16)/ 40.31 # convert to TECU
                dtecstd = dtecstd * 299792458 * freq**2 * (10**-16)/ 40.31
            else:
                print('Invalid observation type specified')
                return None, None
        except Exception as e:
            print(f'Failed to find/process the file for {path.split("/")[-1]} observations')
            print(e)
            return None, None
        finally:
            ds.close()
        
        return dtec, dtecstd

    def obs_epochs(self, path):
        '''This function reads the epochs of geodetic VLBI/VGOS observations and returns the seconds of the day (sod) and day of year (doy).'''

        try:
            t = nc.Dataset(os.path.join(path, 'Observables', 'TimeUTC.nc'))
        except Exception as e:
            print('failed to find/process the TimeUTC.nc file')
            print(e)
            return None, None

        # Extract the epoch of the observation in hours, minutes, and seconds
        hours = t['YMDHM'][:, 3].data
        minutes = t['YMDHM'][:, 4].data
        seconds = t['Second'][:].data

        # Calculate the seconds of the day (sod)
        sod = hours * 3600.0 + minutes * 60 + seconds

        # Extract the year, month, and day
        year = t['YMDHM'][:, 0].data
        month = t['YMDHM'][:, 1].data
        day = t['YMDHM'][:, 2].data

        # Calculate the day of year (doy)
        dates = [datetime(year[i], month[i], day[i]) for i in range(len(year))]
        doy = np.array([date.timetuple().tm_yday for date in dates])

        # create epoch in datetime format
        epochs = [datetime(year[i], month[i], day[i], hours[i], minutes[i], int(seconds[i])) for i in range(len(year))]

        # Close the dataset
        t.close()

        return sod, doy, epochs

    def get_ObsCrossRef(self, session_path):
        ObsCrossRef = nc.Dataset(os.path.join(session_path, 'CrossReference', "ObsCrossRef.nc"), 'r')
        data = {}
        for var_name, var in ObsCrossRef.variables.items():
            data[var_name] = var[:]
        Obs2Scan = data["Obs2Scan"].data
        Obs2Baseline = data["Obs2Baseline"].data
        ObsCrossRef.close()
        return Obs2Scan, Obs2Baseline

    def get_SourceCrossRef(self, session_path):
        SourceCrossRef = nc.Dataset(os.path.join(session_path, 'CrossReference', "SourceCrossRef.nc"), 'r')
        data = {}
        for var_name, var in SourceCrossRef.variables.items():
            data[var_name] = var[:]
        Scan2Source = data["Scan2Source"].data
        CrossRefSourceList = data["CrossRefSourceList"].data
        SourceCrossRef.close()
        return Scan2Source, CrossRefSourceList

    def get_StationCrossRef(self, session_path):
        StationCrossRef = nc.Dataset(os.path.join(session_path, 'CrossReference', "StationCrossRef.nc"), 'r')
        data = {}
        for var_name, var in StationCrossRef.variables.items():
            data[var_name] = var[:]
        NumScansPerStation = data['NumScansPerStation'].data
        CrossRefStationList = data['CrossRefStationList'].data
        Station2Scan = data['Station2Scan'].data
        Scan2Station = data['Scan2Station'].data

        StationCrossRef.close()

        stations = []
        # decode the string back into ASCII
        for i in range(len(CrossRefStationList[:])):
            x = []
            for j in range(len(CrossRefStationList[i, :])):
                x.append(CrossRefStationList[i, j].decode('UTF-8'))
            # concatenate the letters
            stations.append(''.join(x).replace(' ', ''))

        return NumScansPerStation, CrossRefStationList, Station2Scan, Scan2Station, stations

    def get_station_list(self, data_paths):
        stations = []
        for path in data_paths:
            _, _, _, _, sta = self.get_StationCrossRef(path)
            stations += sta
        return stations
    
    def get_station_coords(self, path):
        try:
            Station = nc.Dataset(os.path.join(path, 'Apriori/Station.nc'))
            # extract the station cartesian coordinates (XYZ)
            stationXYZ = Station['AprioriStationXYZ'][:]
            
            # convert the coordinates of the station from XYZ to longitudate, latitude, and altitude
            # create a tranformation object
            transformer = pyproj.Transformer.from_crs({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
                                                    {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'})

            # latitude, longitude, and altitude order
            stationLLA = np.zeros(shape=(len(stationXYZ[:]), len(stationXYZ[0, :])))

            # cartesian(XYZ) to georgraphic(LLA) coordinates
            for n, item in enumerate(stationXYZ[:]):
                # longitude, latitude, and altitude order
                stationLLA[n, 1], stationLLA[n, 0], stationLLA[n, 2] = transformer.transform(item[0], item[1], item[2])
            Station.close()
        except Exception as e1:
            print('failed to find/process the Station.nc file')
            #logger.debug(e1)
        
        return stationLLA

    def get_AzEl(self, session_path, stations, Obs2Scan, Scan2Station, Obs2Baseline):
        # Azel: Az Sta1, El Sta1, Az Sta2, El Sta2
        AzEl_obs = np.zeros((len(Obs2Baseline), 4))
        for sta_index, station in enumerate(stations):
            AzEl = nc.Dataset(os.path.join(session_path, station.upper(), "AzEl.nc"), 'r')
            data = {}
            for var_name, var in AzEl.variables.items():
                data[var_name] = var[:]
            ElTheo = data['ElTheo'][:].data
            AzTheo = data['AzTheo'][:].data
            AzEl.close()

            for i in range(len(Obs2Baseline)):
                if Obs2Baseline[i, 0] == sta_index + 1:
                    AzEl_obs[i, 0] = AzTheo[Scan2Station[Obs2Scan[i]-1, sta_index]-1][0]
                    AzEl_obs[i, 1] = ElTheo[Scan2Station[Obs2Scan[i]-1, sta_index]-1][0]
                elif Obs2Baseline[i, 1] == sta_index + 1:
                    AzEl_obs[i, 2] = AzTheo[Scan2Station[Obs2Scan[i]-1, sta_index]-1][0]
                    AzEl_obs[i, 3] = ElTheo[Scan2Station[Obs2Scan[i]-1, sta_index]-1][0]

        return AzEl_obs
    
    def get_freq(self, path):
        # get the freq in MHz and convert it to Hz
        if os.path.exists(os.path.join(path, 'ObsDerived/EffFreq_bX.nc')):
            ds = nc.Dataset(os.path.join(path, 'ObsDerived/EffFreq_bX.nc'))
            freq = ds['FreqGroupIono'][:].data
        else:
            ds = nc.Dataset(os.path.join(path, 'Observables/RefFreq_bX.nc'))
            freq = ds['RefFreq'][:].data
            freq = freq * 1e6
        ds.close()
        return freq
    
    def get_SNR(self, path):
        dsx = nc.Dataset(os.path.join(path, 'Observables/SNR_bX.nc'))
        s2nrX = dsx['SNR'][:].data
        dsx.close()
        
        try:
            dss = nc.Dataset(os.path.join(path, 'Observables/SNR_bS.nc'))
            s2nrS = dss['SNR'][:].data            
            dss.close()
        except:
            s2nrS = s2nrX

        return s2nrX, s2nrS
    
    def preprocess(self, data, Scan2Source, Obs2Scan):
        # Filter data

        # get the indices of the observations from sources that were scanned more than the minObs 
        """sindx = []
        Obs2Source = Scan2Source[Obs2Scan - 1]
        for i in set(Obs2Source):
            xindx = [j for j in range(len(Obs2Source)) if Obs2Source[j] == i]
            if len(xindx) > 5:
                sindx = sindx + xindx"""
        
        # filter rows by doy
        doyind = [i for i in range(len(data['doy'])) if data['doy'][i] == int(self.doy)]

        # get the indices of the observatins with a signal to noise ration more than snr
        snrindxX = [list(data['s2nrX']).index(data['s2nrX'][i]) for i in range(len(data['s2nrX'])) if data['s2nrX'][i] > 15]
        if len(data['s2nrS'])!=0:
            snrindxS = [list(data['s2nrS']).index(data['s2nrS'][i]) for i in range(len(data['s2nrS'])) if data['s2nrS'][i] > 15]
        else :
            snrindxS = []
        snrind = list(set(snrindxX + snrindxS))

        dtec0ind = [i for i in range(len(data['dtec'])) if data['dtec'][i] != 0]

        # get the indices of the observations with an elevation angle more than the cut off angle   
        elind = np.where(
            (np.rad2deg(data['El_sta1']) > 15) & (np.rad2deg(data['El_sta2']) > 15)
        )[0].tolist()

        # get the indices of observations with non-zero standard deviation
        # obs. w. zero sta. dev. are basically made with the twin telescopes, i.e., Onsala13SW and Onsala13NE, as a baseline
        std0ind = [i for i in range(len(data['dtecstd'])) if data['dtecstd'][i] != 0]

        # find the common indices
        indx = np.intersect1d(dtec0ind, np.intersect1d(snrind, np.intersect1d(elind, np.intersect1d(std0ind, doyind))))
        data = data.iloc[indx]

        # Normalize temporal features
        data.loc[:, 'sin_utc'] = np.sin(data['sod'] / 86400 * 2 * np.pi)
        data.loc[:, 'cos_utc'] = np.cos(data['sod'] / 86400 * 2 * np.pi)
        data.loc[:, 'sod_normalize'] = 2 * data['sod'] / 86400 - 1

        return data

    def coord_tranform(self, lats, lons, epochs):
        n = lats.shape[0]
        coords = np.zeros((n, 3))
        for i in range(n):
            coords[i] = [1+450/6371, lats[i], lons[i]]
        coords = list(coords)
        coord_inp = Coords(coords, 'GEO', 'sph')
        coord_inp.ticks = Ticktock(epochs, 'UTC')
        
        coord_out = coord_inp.convert('SM', 'sph')
        return coord_out

    def add_ipp(self, data):
        # calculate the IPPs
        # mean radius of earth and ionospheric layer height
        R, h = 6371.0, 450

        for sta in ['1', '2']:
            # Bull. Geod. Sci, Articles Section, Curitiba, v. 23, no4, p.669 - 683, Oct - Dec, 2017.
            # calculate the Earth-centred angle                      
            Elev = data[f'El_sta{sta}']
            Psi = np.pi/2 - Elev - np.arcsin(R/(R+h)*np.cos(Elev))

            # compute the latitude of the IPP
            Az = data[f'Az_sta{sta}']
            lat = np.deg2rad(data[f'sta{sta}_lat'])      
            Phi = np.arcsin(np.sin(lat)*np.cos(Psi) + np.cos(lat)*np.sin(Psi)*np.cos(Az))

            # compute the longitude of the IPP
            lon = np.deg2rad(data[f'sta{sta}_lon'])
            Lambda = lon + np.arcsin(np.sin(Psi)*np.sin(Az)/np.cos(Phi))

            # save the latitude and the longitude of the ionospheric points
            data[f'IPP_sta{sta}_lat'] = Phi
            data[f'IPP_sta{sta}_lon'] = Lambda

            coords_sm = self.coord_tranform(np.rad2deg(Phi.tolist()), np.rad2deg(Lambda.tolist()), data['epoch'].tolist())
            data[f'IPP_sta{sta}_smlat'] = coords_sm.lati.astype(np.float32)
            data[f'IPP_sta{sta}_smlon'] = coords_sm.long.astype(np.float32)

        return data

    def load_data(self, data_files):
        data = pd.DataFrame()
        if len(data_files) == 0:
            return data
        for path in data_files:
            if not os.path.exists(os.path.join(path, 'Apriori', 'Station.nc')):
                continue
            current_data = pd.DataFrame()
            dtec, dtecstd = self.read_dTEC(path)
            sod, doy, epochs = self.obs_epochs(path)
            Obs2Scan, Obs2Baseline = self.get_ObsCrossRef(path)
            Scan2Source, CrossRefSourceList = self.get_SourceCrossRef(path)
            NumScansPerStation, CrossRefStationList, Station2Scan, Scan2Station, stations = self.get_StationCrossRef(path)
            station_coords = self.get_station_coords(path)
            AzEl = self.get_AzEl(path, stations, Obs2Scan,Scan2Station, Obs2Baseline)
            s2nrX, s2nrS = self.get_SNR(path)

            # create dataframe with columns:    dtec, dtecstd, doy, sod, 
            #                                   sta1, sta1_ind, sta1_lat, sta1_lon, Az_sta1, El_sta1
            #                                   sta2, sta2_ind, sta2_lat, sta2_lon, Az_sta2, El_sta2
            #                                   s2nrX, s2nrS  
            current_data['dtec'] = dtec
            current_data['dtecstd'] = dtecstd
            current_data['doy'] = doy
            current_data['sod'] = sod
            current_data['epoch'] = epochs
            current_data['sta1'] = [stations[Obs2Baseline[i, 0]-1] for i in range(len(Obs2Scan))]
            current_data['sta1_ind'] = [Obs2Baseline[i, 0]-1 for i in range(len(Obs2Scan))]
            current_data['sta1_lat'] = [station_coords[Obs2Baseline[i, 0]-1, 1] for i in range(len(Obs2Scan))]
            current_data['sta1_lon'] = [station_coords[Obs2Baseline[i, 0]-1, 0] for i in range(len(Obs2Scan))]
            current_data['Az_sta1'] = [AzEl[i, 0] for i in range(len(Obs2Baseline))]
            current_data['El_sta1'] = [AzEl[i, 1] for i in range(len(Obs2Baseline))]
            current_data['sta2'] = [stations[Obs2Baseline[i, 1]-1] for i in range(len(Obs2Scan))]
            current_data['sta2_ind'] = [Obs2Baseline[i, 1]-1 for i in range(len(Obs2Scan))]
            current_data['sta2_lat'] = [station_coords[Obs2Baseline[i, 1]-1, 1] for i in range(len(Obs2Scan))]
            current_data['sta2_lon'] = [station_coords[Obs2Baseline[i, 1]-1, 0] for i in range(len(Obs2Scan))]
            current_data['Az_sta2'] = [AzEl[i, 2] for i in range(len(Obs2Baseline))]
            current_data['El_sta2'] = [AzEl[i, 3] for i in range(len(Obs2Baseline))]
            current_data['s2nrX'] = s2nrX
            current_data['s2nrS'] = s2nrS

            data = pd.concat([data, current_data], ignore_index=True)

        data = self.preprocess(data, Scan2Source, Obs2Scan)

        # add IPP here
        data = self.add_ipp(data)

        columns_to_keep = ['dtec', 'IPP_sta1_smlat', 'IPP_sta1_smlon', 'IPP_sta2_smlat', 'IPP_sta2_smlon', 'sin_utc', 'cos_utc', 'sod_normalize', 'sta1_ind', 'sta2_ind', 'El_sta1', 'El_sta2']
        return torch.tensor(data[columns_to_keep].values, dtype=torch.float32)
    
    def __getitem__(self, idx):
        x1 = self.data[idx, [1,2,5,6,7,8,10]]
        x2 = self.data[idx, [3,4,5,6,7,9,11]]
        y = self.data[idx, 0]
        tech = torch.tensor(1, dtype=torch.int64)  # 1 == VLBI
        return x1, x2, y, tech

class FusionDataset(Dataset):
    def __init__(self, gnss_dataset, vlbi_dataset):
        self.gnss_dataset = gnss_dataset
        self.vlbi_dataset = vlbi_dataset
        self.total_len = len(gnss_dataset) + len(vlbi_dataset)
        self.indices = list(range(self.total_len))

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < len(self.gnss_dataset):
            x, y, tech = self.gnss_dataset[idx]
        else:
            x, y, tech = self.vlbi_dataset[idx - len(self.gnss_dataset)]
        return x, y, tech
    
class FusionDTECDataset(Dataset):
    def __init__(self, gnss_dataset, vlbi_dataset):
        self.gnss_dataset = gnss_dataset
        self.vlbi_dataset = vlbi_dataset
        self.total_len = len(gnss_dataset) + len(vlbi_dataset)
        self.indices = list(range(self.total_len))
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if idx < len(self.gnss_dataset):
            x, y, tech = self.gnss_dataset[idx]
            return x, y, tech 
        else:
            x1, x2, y, tech = self.vlbi_dataset[idx - len(self.gnss_dataset)]
            return (x1, x2), y, tech
        

def get_GNSS_data(config):
    test_split = config['training']['test_size']
    
    if config["preprocessing"]["split"] == 'lists':
        train_dataset = SingleGNSSDataset(config, split='train')
        val_dataset = SingleGNSSDataset(config, split='val')
        test_dataset = SingleGNSSDataset(config, split='test')
    else:
        dataset = SingleGNSSDataset(config, split=config['preprocessing']['split'])
        
        # Ensure splits sum to 1
        train_val_split = 1 - test_split
        val_split = 0.2 * train_val_split
        total_size = len(dataset)
        val_size = int(val_split * total_size)
        test_size = int(test_split * total_size)
        train_size = total_size - val_size - test_size

        # Split the dataset into train, validation, and test sets
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    return train_dataset, val_dataset, test_dataset

def get_VLBI_data(config):
    
    train_dataset = SingleVLBIDataset(config, split="train")
    val_dataset = SingleVLBIDataset(config, split="val")
    test_dataset = SingleVLBIDataset(config, split="test")

    return train_dataset, val_dataset, test_dataset

def get_VLBI_dtec_data(config):
    test_split = config['training']['test_size']
    dataset = DTECVLBIDataset(config, split='')

    if config["preprocessing"]["split"] == 'lists':
        total_size = len(dataset)
        val_size = 0
        test_size = 0
        train_size = total_size - val_size - test_size
    else:
        train_val_split = 1 - test_split
        val_split = 0.2 * train_val_split
        total_size = len(dataset)
        val_size = int(val_split * total_size)
        test_size = int(test_split * total_size)
        train_size = total_size - val_size - test_size

    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset

class CollateWithSH:
    def __init__(self, sh_degree, sh_encoding):
        self.sh_degree = sh_degree
        self.sh_encoding = sh_encoding
        if self.sh_encoding:
            self.sh_encoder = SphericalHarmonics(legendre_polys=sh_degree)

    def __call__(self, batch):
        xs, ys, techs = zip(*batch)
        xs = torch.stack(xs)
        ys = torch.stack(ys)
        techs = torch.stack(techs)

        if self.sh_encoding:
            # Extract latitude and longitude
            sm_lat = xs[:, 0]
            sm_lon = xs[:, 1]
            other_features = xs[:, 2:]

            # Stack lat and lon for the batch
            lonlat = torch.stack((sm_lon, sm_lat), dim=-1)

            # Compute embeddings for the batch
            embeddings = self.sh_encoder(lonlat)  # Ensure sh_encoder accepts batch inputs

            # Concatenate embeddings with other features
            x = torch.cat([embeddings, other_features], dim=1)
        else:
            # If SH_encoding is False, use the features as they are
            xs[:, 0] = (xs[:, 0] - (-90)) / (90 - (-90)) * 2 - 1
            xs[:, 1] = (xs[:, 1] - (-180)) / (180 - (-180)) * 2 - 1
            x = xs  # xs already contains all features

        return x, ys, techs

class CollateDTEC:
    def __init__(self, sh_degree, sh_encoding):
        self.sh_degree = sh_degree
        self.sh_encoding = sh_encoding
        if self.sh_encoding:
            self.sh_encoder = SphericalHarmonics(legendre_polys=sh_degree)

    def __call__(self, batch):
        gnss_batch = [item for item in batch if isinstance(item[0], torch.Tensor)]
        vlbi_batch = [item for item in batch if isinstance(item[0], tuple)]

        if gnss_batch:
            gnss_xs, gnss_ys, gnss_techs = zip(*gnss_batch)
            gnss_xs = torch.stack(gnss_xs)
            gnss_ys = torch.stack(gnss_ys)
            gnss_techs = torch.stack(gnss_techs)

            if self.sh_encoding:
                sm_lat = gnss_xs[:, 0]
                sm_lon = gnss_xs[:, 1]
                other_features = gnss_xs[:, 2:]
                lonlat = torch.stack((sm_lon, sm_lat), dim=-1)
                embeddings = self.sh_encoder(lonlat)
                gnss_xs = torch.cat([embeddings, other_features], dim=1)

        if vlbi_batch:
            vlbi_xs1, vlbi_xs2, vlbi_ys, vlbi_techs = zip(*[(x1, x2, y, tech) for (x1, x2), y, tech in vlbi_batch])
            vlbi_xs1 = torch.stack(vlbi_xs1)
            vlbi_xs2 = torch.stack(vlbi_xs2)
            vlbi_ys = torch.stack(vlbi_ys)
            vlbi_techs = torch.stack(vlbi_techs)

            if self.sh_encoding:
                sm_lat1 = vlbi_xs1[:, 0]
                sm_lon1 = vlbi_xs1[:, 1]
                other_features1 = vlbi_xs1[:, 2:]
                lonlat1 = torch.stack((sm_lon1, sm_lat1), dim=-1)
                embeddings1 = self.sh_encoder(lonlat1)
                vlbi_xs1 = torch.cat([embeddings1, other_features1], dim=1)

                sm_lat2 = vlbi_xs2[:, 0]
                sm_lon2 = vlbi_xs2[:, 1]
                other_features2 = vlbi_xs2[:, 2:]
                lonlat2 = torch.stack((sm_lon2, sm_lat2), dim=-1)
                embeddings2 = self.sh_encoder(lonlat2)
                vlbi_xs2 = torch.cat([embeddings2, other_features2], dim=1)

            # Interleave vlbi_xs1 and vlbi_xs2
            vlbi_xs = torch.empty((vlbi_xs1.size(0) * 2, vlbi_xs1.size(1)), dtype=vlbi_xs1.dtype)
            vlbi_xs[0::2] = vlbi_xs1
            vlbi_xs[1::2] = vlbi_xs2
            vlbi_y = torch.empty((vlbi_ys.size(0) * 2), dtype=vlbi_ys.dtype)
            vlbi_y[0::2] = vlbi_ys
            vlbi_y[1::2] = vlbi_ys
            vlbi_tech = torch.empty((vlbi_techs.size(0) * 2), dtype=vlbi_techs.dtype)
            vlbi_tech[0::2] = vlbi_techs
            vlbi_tech[1::2] = vlbi_techs

        if gnss_batch and vlbi_batch:
            xs = torch.cat([gnss_xs, vlbi_xs], dim=0)
            ys = torch.cat([gnss_ys, vlbi_y], dim=0)
            techs = torch.cat([gnss_techs, vlbi_tech], dim=0)
        elif gnss_batch:
            xs, ys, techs = gnss_xs, gnss_ys, gnss_techs
        else:
            xs, ys, techs = vlbi_xs, vlbi_y, vlbi_tech

        return xs, ys, techs

def get_data_loaders(config):
    batch_size = config['training']['batchsize']
    shuffle = config["data"]["shuffle"]
    sh_degree = config["preprocessing"]["SH_degree"]  # Get sh_degree from config
    sh_encoding = config["preprocessing"]["SH_encoding"]  # Get SH_encoding flag


    # Create an instance of the collate function with sh_degree
    if config["data"]["mode"] == "DTEC_Fusion":
        collate_fn = CollateDTEC(sh_degree, sh_encoding)
    else:
        collate_fn = CollateWithSH(sh_degree, sh_encoding)

    if config["data"]["mode"] == "GNSS":
        train_dataset, val_dataset, test_dataset = get_GNSS_data(config)
        test_dataset = FusionDataset(test_dataset, SingleVLBIDataset(config, split="test"))

    elif config["data"]["mode"] == "Fusion":
        train_vlbi, val_vlbi, test_vlbi = get_VLBI_data(config)
        train_gnss, val_gnss, test_gnss = get_GNSS_data(config)

        train_dataset = FusionDataset(train_gnss, train_vlbi)
        val_dataset = FusionDataset(val_gnss, val_vlbi)
        test_dataset = FusionDataset(test_gnss, test_vlbi)

    elif config["data"]["mode"] == "DTEC_Fusion":
        train_gnss, val_gnss, test_gnss = get_GNSS_data(config)
        train_vlbi, val_vlbi, test_vlbi = get_VLBI_dtec_data(config)

        # make fusion dataset here somehow
        train_dataset = FusionDTECDataset(train_gnss, train_vlbi)
        val_dataset = FusionDTECDataset(val_gnss, val_vlbi)
        test_dataset = FusionDTECDataset(test_gnss, test_vlbi)

    if config["training"]["vlbi_sampling_weight"] != 1.0 and (config["data"]["mode"] == "Fusion" or config["data"]["mode"] == "DTEC_Fusion"):
        # Create weights based on the technique
        vlbi_weight = config["training"]["vlbi_sampling_weight"]
        total_len = len(train_dataset)
        weights = []
        for idx in range(total_len):
            _, _, tech = train_dataset[idx]
            weight = vlbi_weight if tech == 1 else 1.0
            weights.append(weight)
        weights = torch.tensor(weights)
        sampler = WeightedRandomSampler(weights, len(weights))

        # Create data loaders with custom collate_fn
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=config["training"]["num_workers"], collate_fn=collate_fn)
    else:
        # Create data loaders with custom collate_fn
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=config["training"]["num_workers"], collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=config["training"]["num_workers"], collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=config["training"]["num_workers"], collate_fn=collate_fn)

    return train_loader, val_loader, test_loader

def get_stations(config):
    dataset = DTECVLBIDataset(config, split='')
    return dataset.stations
