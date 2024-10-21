import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Example dataset class, modify it to suit your data
class GNSSVLBIDataLoader(Dataset):
    def __init__(self, data_file):
        # Load and preprocess your data here
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        # Add logic to load your data from file (CSV, HDF5, etc.)
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return data and label, modify based on your data format
        x = self.data[idx]['input']
        y = self.data[idx]['label']
        return x, y

def get_data_loaders(data_file, batch_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True):
    dataset = GNSSVLBIDataLoader(data_file)
    
    # Ensure splits sum to 1
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
