import os
import h5py
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from utils.locationencoder.pe import SphericalHarmonics

class MultiGNSSDataLoader(Dataset):
    def __init__(self, config, h5_files):
        self.h5_files = h5_files  # List of paths to HDF5 files
        self.columns_to_load = config["data"]["columns_to_load"]
        self.file_data_index = self._create_file_data_index()
        self.sh_encoder = SphericalHarmonics(legendre_polys=16)  # Spherical Harmonics Encoder

    def _create_file_data_index(self):
        """Create an index to keep track of the start and end index of data in each file."""
        file_data_index = []
        total_data_count = 0
        for file in self.h5_files:
            with h5py.File(file, 'r') as h5_file:
                file_size = len(h5_file[self.columns_to_load[0]])  # Assuming all columns have the same length
                file_data_index.append((file, total_data_count, total_data_count + file_size))
                total_data_count += file_size
        return file_data_index

    def _find_file_and_index(self, idx):
        """Find which file and index corresponds to a global dataset index."""
        for file, start_idx, end_idx in self.file_data_index:
            if start_idx <= idx < end_idx:
                return file, idx - start_idx
        raise IndexError(f"Index {idx} is out of bounds.")

    def __len__(self):
        return sum([end - start for _, start, end in self.file_data_index])

    def __getitem__(self, idx):
        # Find the correct file and the local index in that file
        file, local_idx = self._find_file_and_index(idx)
        
        # Load only the required row from the appropriate file
        with h5py.File(file, 'r') as h5_file:
            data_row = {col: h5_file[col][local_idx] for col in self.columns_to_load}
            df_row = pd.DataFrame(data_row, index=[0])
            df_row = self.preprocess(df_row)  # Preprocess just this row
            x = df_row[['sm_lat', 'sm_lon', 'sod']].values.flatten()
            y = df_row['vtec'].values[0]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def preprocess(self, df_row):
        # Apply the preprocessing logic here
        pass


class SingleGNSSDataLoader_memoryerror(Dataset):
    def __init__(self, config):
        # Load and preprocess your data here
        year = config['year']
        doy = config['doy']
        self.elev = config['data']['elevation']  # Elevation threshold
        self.columns_to_load = config["data"]["columns_to_load"]
        self.sh_encoder = SphericalHarmonics(legendre_polys=16)  # Spherical Harmonics Encoder
        
        data_file = os.path.join(config['data']['GNSS_data_path'], str(year), str(doy), f'ccl_{year}{doy}_30_5.h5')
        self.data = self.load_data(data_file)
    
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
        # Filter data
        mask = (abs(df['dcbs']) > 1e-3) & (abs(df['dcbr']) > 1e-3) & (df['vtec'] > 2.0) & \
            (df['vtec'] <= 200) & (df['satele'] >= self.elev)
        df = df[mask]
        
        # Handle empty dataframe case
        if df.empty:
            raise ValueError("DataFrame is empty after filtering, check your filtering conditions or data.")

        return df
    
    def preprocess(self, df):
        # Add time features and normalize
        df.loc[:, 'sin_utc'] = np.sin(df['sod'] / 86400 * 2 * np.pi) # sin encoding of time (repeated each day)
        df.loc[:, 'cos_utc'] = np.cos(df['sod'] / 86400 * 2 * np.pi) # cos encoding of time (repeated each day)
        df.loc[:, 'sod_normalize'] = 2 * df['sod'] / 86400 - 1 # Normalize to [-1, 1]

        # Normalize spatial and time features
        df.loc[:, 'sm_lon'] = (df['sm_lon'] + 180) % 360 - 180  # modulo to [-180, 180]
        df.loc[:, 'sm_lat'] = (df['sm_lat'] - (-90)) / (90 - (-90)) * 2 - 1  # Normalize to [-1, 1]
        df.loc[:, 'sm_lon'] = (df['sm_lon'] - (-180)) / (180 - (-180)) * 2 - 1  # Normalize to [-1, 1]

        # Spherical Harmonic Positional Encoding
        lat = torch.tensor(df['sm_lat'].values)
        lon = torch.tensor(df['sm_lon'].values)
        lonlat = torch.stack((lon, lat), dim=-1)
        embedded_lonlat = np.array(self.sh_encoder(lonlat))  # Apply SH encoding

        # Add positional encodings to the dataset
        df['embedded_lonlat'] = embedded_lonlat.tolist()

        return df
    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Use iloc for DataFrame access by index
        row = self.data.iloc[idx]
        #######################################################to tensor?
        row = self.preprocess(row)
        
        # Extract features and label
        x = torch.tensor(row[['embedded_lonlat', 'sin_utc', 'cos_utc', 'sod_normalize']].values.flatten(), dtype=torch.float32)
        y = torch.tensor(row['vtec'], dtype=torch.float32)
        
        return x, y
    
class SingleGNSSDataLoader(Dataset):
    def __init__(self, config):
        # Store file path and column info for lazy loading
        year = config['year']
        doy = config['doy']
        self.elev = config['data']['elevation']  # Elevation threshold
        self.columns_to_load = config["data"]["columns_to_load"]
        self.sh_encoder = SphericalHarmonics(legendre_polys=16)  # Spherical Harmonics Encoder
        
        self.data_file = os.path.join(config['data']['GNSS_data_path'], str(year), str(doy), f'ccl_{year}{doy}_30_5.h5')
        
        # Get dataset size without loading the whole data
        with h5py.File(self.data_file, 'r') as h5_file:
            # If no specific columns are provided, load all columns
            if self.columns_to_load is None:
                self.columns_to_load = list(h5_file.keys())  # Load all column names
            self.dataset_size = len(h5_file[self.columns_to_load[0]])  # Assuming all columns have the same size
    
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Load the required row from the file on-demand
        with h5py.File(self.data_file, 'r') as h5_file:
            data_row = {col: h5_file[col][idx] for col in self.columns_to_load}
            df_row = pd.DataFrame(data_row, index=[0])

        # Preprocess the data row
        features = self.preprocess(df_row)
                
        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(df_row['vtec'].values[0], dtype=torch.float32)
        
        return x, y

    
    def preprocess_item(self, df):

        df.loc[:, 'sin_utc'] = np.sin(df['sod'] / 86400 * 2 * np.pi)
        df.loc[:, 'cos_utc'] = np.cos(df['sod'] / 86400 * 2 * np.pi)
        df.loc[:, 'sod_normalize'] = 2 * df['sod'] / 86400 - 1

        df.loc[:, 'sm_lon'] = (df['sm_lon'] + 180) % 360 - 180
        df.loc[:, 'sm_lon'] = (df['sm_lon'] - (-180)) / (180 - (-180)) * 2 - 1
        df.loc[:, 'sm_lat'] = (df['sm_lat'] - (-90)) / (90 - (-90)) * 2 - 1

        lat = torch.tensor(df['sm_lat'].values)
        lon = torch.tensor(df['sm_lon'].values)
        lonlat = torch.stack((lon, lat), dim=-1)
        embedded_lonlat = np.array(self.sh_encoder(lonlat))

        # Extract other features (e.g., sin_utc, cos_utc, sod_normalize)
        other_features = df[['sin_utc', 'cos_utc', 'sod_normalize']].values

        # Combine positional encodings with other features
        combined_features = np.hstack((embedded_lonlat, other_features))

        #for i in range(embedded_lonlat.shape[1]):
        #    df[f'embedded_lonlat_{i}'] = embedded_lonlat[:, i]
        
        return combined_features


def get_data_loaders(config):
    batch_size = config['training']['batchsize']
    test_split = config['training']['test_size']
    shuffle = config["data"]["shuffle"]
    
    dataset = SingleGNSSDataLoader_memoryerror(config)
    
    # Ensure splits sum to 1
    train_val_split = 1 - test_split
    val_split = 0.2 * train_val_split
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)
    train_size = total_size - val_size - test_size

    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=config["training"]["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config["training"]["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=config["training"]["num_workers"])
    
    return train_loader, val_loader, test_loader
