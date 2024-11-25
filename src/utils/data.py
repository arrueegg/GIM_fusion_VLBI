import os
import h5py
import pandas as pd
import torch
import numpy as np
from io import StringIO
from datetime import datetime, timedelta
from spacepy.coordinates import Coords
from spacepy.time import Ticktock
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from utils.locationencoder.pe import SphericalHarmonics

class SingleGNSSDataset(Dataset):
    def __init__(self, config, split='random'):
        # Load and preprocess your data here
        year = config['year']
        doy = config['doy']
        self.elev = config['preprocessing']['elevation']  # Elevation threshold
        self.columns_to_load = config["data"]["columns_to_load"]
        self.sh_encoding = config["preprocessing"]["SH_encoding"]
        # Initialize Spherical Harmonics Encoder if required
        if self.sh_encoding:
            self.sh_degree = config["preprocessing"]["SH_degree"]
            self.sh_encoder = SphericalHarmonics(legendre_polys=self.sh_degree)
        
        data_file = os.path.join(config['data']['GNSS_data_path'], str(year), str(doy), f'ccl_{year}{doy}_30_5.h5')
        self.split = split
        self.mode = config['data']['mode']
        self.data = self.load_data(data_file)
    
    def __len__(self):
        return len(self.data)

    def load_data(self, data_file):
        data = {}

        with h5py.File(data_file, 'r') as h5_file:
            # If no specific columns are provided, load all columns
            if self.columns_to_load is None:
                self.columns_to_load = list(h5_file.keys())  # Load all column names

            # Load the data for the specified columns
            for column in self.columns_to_load:
                data[column] = h5_file[column][:]
        
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(data)

        df = self.filter_df(df)

        return df
    
    def filter_df(self, df):
        
        df['station'] = df['station'].apply(lambda x: x.decode('utf-8').upper() if isinstance(x, bytes) else x)

        if self.split != 'random':
            sta_list = np.loadtxt(f'./src/data_processing/sit_{self.split}.list', dtype=str)
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
        df.loc[:, 'sm_lon'] = (df['sm_lon'] + 180) % 360 - 180

        # Precompute SH embeddings if enabled
        if self.sh_encoding:
            latitudes = torch.tensor(df['sm_lat'].values)
            longitudes = torch.tensor(df['sm_lon'].values)
            lonlat = torch.stack((longitudes, latitudes), dim=-1)
            
            # Compute embeddings for each row and add to DataFrame
            embeddings = self.sh_encoder(lonlat)
            embedding_cols = pd.DataFrame(embeddings, index=df.index)
            df = pd.concat([df, embedding_cols], axis=1)
        else:
            df.loc[:, 'sm_lat'] = (df['sm_lat'] - (-90)) / (90 - (-90)) * 2 - 1
            df.loc[:, 'sm_lon'] = (df['sm_lon'] - (-180)) / (180 - (-180)) * 2 - 1
        
        #return df
        columns_to_keep = ['vtec']
    
        if self.sh_encoding:
            columns_to_keep.extend(df.columns[-self.sh_degree**2:])  # SH embedding columns
        else:
            columns_to_keep.extend(['sm_lat', 'sm_lon'])  # `sm_lat` and `sm_lon` if not using SH

        columns_to_keep.extend(['sin_utc', 'cos_utc', 'sod_normalize'])
        
        return torch.tensor(df[columns_to_keep].values, dtype=torch.float32)

    def __getitem__(self, idx):

        x = self.data[idx, 1:]  # All columns except the last one as features
        y = self.data[idx, 0]   # Last column as the label

        tech = torch.tensor(0, dtype=torch.int64)  # 0 == GNSS

        return x, y, tech
    
class SingleVLBIDataset(Dataset):
    def __init__(self, config, split='random'):
        # Load and preprocess your data here
        self.year = config['year']
        self.doy = config['doy']
        self.vlbi_path = config['data']['VLBI_data_path']
        self.sh_encoding = config["preprocessing"]["SH_encoding"]
        # Initialize Spherical Harmonics Encoder if required
        if self.sh_encoding:
            self.sh_degree = config["preprocessing"]["SH_degree"]
            self.sh_encoder = SphericalHarmonics(legendre_polys=self.sh_degree)

        data_files = self.get_file_path(config)
        self.split = split
        self.data = self.load_data(data_files)
    
    def __len__(self):
        return len(self.data)

    def get_file_path(self, config):
        
        date1 = datetime(int(self.year), 1, 1) + timedelta(days=int(self.doy) - 1)
        date2 = date1 - timedelta(days=1)
        name1 = date1.strftime('%y%b%d').upper()
        name2 = date2.strftime('%y%b%d').upper()
        vlbi_path = os.path.join(config["data"]["VLBI_data_path"], "VLBI", f"{self.year}")
        vgos_path = os.path.join(config["data"]["VLBI_data_path"], "VGOS", f"{self.year}")

        paths = []
        for _, dirs, _ in os.walk(vlbi_path):
            for dir_name in dirs:
                if dir_name.startswith(name1) or dir_name.startswith(name2): 
                    paths.append(os.path.join(vlbi_path, dir_name, dir_name+'.txt'))  
        
        for _, dirs, _ in os.walk(vgos_path):
            for dir_name in dirs:
                if dir_name.startswith(name1) or dir_name.startswith(name2):  
                    paths.append(os.path.join(vgos_path, dir_name, dir_name+'.txt'))

        return paths

    def load_data(self, data_files):
        data = pd.DataFrame()
        if len(data_files) == 0:
            return data
        data_list = [self.load_txt(file) for file in data_files]
        data = pd.concat(data_list, ignore_index=True)

        data = self.preprocess(data)
        return data

    def load_txt(self, path):
        """
        Load a text file of session results and process its contents into separate DataFrames.

        Parameters:
            path (str): The path to the directory containing the text file.

        Returns:
            dict: A dictionary containing DataFrames for each table in the text file.
                - 'df_vtecs' (DataFrame): DataFrame for the 'vtecs' table.
                - 'df_biases' (DataFrame): DataFrame for the 'biases' table.
                - 'df_gradients' (DataFrame): DataFrame for the 'gradients' table.
                - 'df_instr_offsets' (DataFrame): DataFrame for the 'instr_offsets' table.
        """
        # Read the text file
        with open(path, 'r') as file:
            data = file.read()

        # Split the text into tables based on empty lines
        tables = data.split('\n\n')

        # Define column names for each table
        column_names = [
            ['station', 'date', 'epoch', 'vgos_vtec', 'v_vtec_sigma', 'gims_vtec', 'madr_vtec'],
            ['station', 'bias w.r.t. GIMs', 'bias w.r.t. SMTMs'],
            ['station', 'date', 'Gn', 'Gn_sigma', 'Gs', 'Gs_sigma'],
            ['station', 'date', 'instr_offset', 'io_sigma']
        ]

        df_names = ['vtecs', 'biases', 'gradients', 'instr_offsets']

        dfs = {}
        # Process each table
        for i, table in enumerate(tables):
            # Skip empty tables
            if not table.strip():
                continue
            
            # Use StringIO to simulate a file for pandas
            table_data = StringIO(table)
            
            # Read the table as a DataFrame with predefined column names
            df = pd.read_csv(table_data, sep='\s+', skipinitialspace=True, names=column_names[i], skiprows=1)

            # Store the DataFrame in the dictionary with a meaningful key
            dfs[f'df_{df_names[i]}'] = df

        return dfs['df_vtecs']

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

        sta_coords = pd.read_json(os.path.join(self.vlbi_path, "station_coords.json"))
        
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

        # Filter data
        mask = (df['vtec'] > 2.0) & (df['vtec'] <= 200)
        df = df[mask]
        
        # Handle empty dataframe case
        if df.empty:
            return df

        df.loc[:, 'sin_utc'] = np.sin(df['sod'] / 86400 * 2 * np.pi)
        df.loc[:, 'cos_utc'] = np.cos(df['sod'] / 86400 * 2 * np.pi)
        df.loc[:, 'sod_normalize'] = 2 * df['sod'] / 86400 - 1

        # Precompute SH embeddings if enabled
        if self.sh_encoding:
            latitudes = torch.tensor(df['sm_lat'].values)
            longitudes = torch.tensor(df['sm_lon'].values)
            lonlat = torch.stack((longitudes, latitudes), dim=-1)
            
            # Compute embeddings for each row and add to DataFrame
            embeddings = self.sh_encoder(lonlat)
            embedding_cols = pd.DataFrame(embeddings, index=df.index)
            df = pd.concat([df, embedding_cols], axis=1)
        else:
            df.loc[:, 'sm_lat'] = (df['sm_lat'] - (-90)) / (90 - (-90)) * 2 - 1
            df.loc[:, 'sm_lon'] = (df['sm_lon'] - (-180)) / (180 - (-180)) * 2 - 1
        
        columns_to_keep = ['vtec']
    
        if self.sh_encoding:
            columns_to_keep.extend(df.columns[-self.sh_degree**2:])  # SH embedding columns
        else:
            columns_to_keep.extend(['sm_lat', 'sm_lon'])  # `sm_lat` and `sm_lon` if not using SH

        columns_to_keep.extend(['sin_utc', 'cos_utc', 'sod_normalize'])
        return torch.tensor(df[columns_to_keep].values, dtype=torch.float32)
        
    def coord_transform(self, coords, epochs, inp_type, out_type):
        coord_inp = Coords(coords, inp_type, 'sph')
        coord_inp.ticks = Ticktock(epochs, 'UTC')
        return coord_inp.convert(out_type, 'sph')

    def __getitem__(self, idx):

        x = self.data[idx, 1:]  # All columns except the last one as features
        y = self.data[idx, 0]   # Last column as the label
        
        tech = torch.tensor(1, dtype=torch.int64)  # 1 == VLBI

        return x, y, tech
    
class FusionDataset(Dataset):
    def __init__(self, gnss_dataset, vlbi_dataset):
        self.gnss_dataset = gnss_dataset
        self.vlbi_dataset = vlbi_dataset
        self.data = self.combine()

    def __len__(self):
        return len(self.data)

    def combine(self):
        # Add "technique" column to the GNSS dataset
        gnss_technique_col = torch.zeros((self.gnss_dataset.data.size(0), 1), dtype=torch.float32)
        gnss_tensor = torch.cat([self.gnss_dataset.data, gnss_technique_col], dim=1)

        # Check if VLBI dataset is available and not empty
        if hasattr(self.vlbi_dataset, 'data') and self.vlbi_dataset.data is not None and self.vlbi_dataset.data.size(0) > 0:
            # Add "technique" column to the VLBI dataset
            vlbi_technique_col = torch.ones((self.vlbi_dataset.data.size(0), 1), dtype=torch.float32)
            vlbi_tensor = torch.cat([self.vlbi_dataset.data, vlbi_technique_col], dim=1)

            # Concatenate GNSS and VLBI tensors
            combined_tensor = torch.cat([gnss_tensor, vlbi_tensor], dim=0)
            return combined_tensor
        
        # If VLBI dataset is empty, return GNSS tensor only
        return gnss_tensor

    def __getitem__(self, idx):
        # Split features and label from the combined tensor
        tech = self.data[idx, -1]
        x = self.data[idx, 1:-1]  # All columns except the last one as features
        y = self.data[idx, 0]   # Last column as the label
        return x, y, tech

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

def get_data_loaders(config):
    batch_size = config['training']['batchsize']
    shuffle = config["data"]["shuffle"]

    if config["data"]["mode"] == "GNSS":
        train_dataset, val_dataset, test_dataset = get_GNSS_data(config)
        test_dataset = FusionDataset(test_dataset, SingleVLBIDataset(config, split="test"))
    
    elif config["data"]["mode"] == "Fusion":
        train_gnss, val_gnss, test_gnss = get_GNSS_data(config)
        train_vlbi, val_vlbi, test_vlbi = get_VLBI_data(config)
        
        train_dataset = FusionDataset(train_gnss, train_vlbi)
        val_dataset = FusionDataset(val_gnss, val_vlbi)
        test_dataset = FusionDataset(test_gnss, test_vlbi)

    if config["training"]["vlbi_sampling_weight"] != 1.0 and config["data"]["mode"] == "Fusion":
        # Assuming `train_dataset.data` is a tensor and the last column is the "technique" indicator
        technique_column = train_dataset.data[:, -1]  # Extract the technique column
        # Create weights: 1.0 for VLBI (technique == 1), 0.005 for GNSS (technique == 0)
        weights = torch.where(technique_column == 1, config["training"]["vlbi_sampling_weight"], 1.0)
        sampler = WeightedRandomSampler(weights, len(weights))

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=config["training"]["num_workers"])
    else:
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=config["training"]["num_workers"])
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config["training"]["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=config["training"]["num_workers"])
    
    return train_loader, val_loader, test_loader
