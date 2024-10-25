"""
Author: Arno Rüegg
Date: 2024-10-21
Description: Global Ionospheric Maps from GNSS and VLBI data
"""

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import logging
import wandb
import os
from tqdm import tqdm

from models.model import get_model, init_xavier
from utils.loss_function import get_criterion
from utils.optimizers import get_optimizer
from utils.config_parser import parse_config
from utils.data import get_data_loaders

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
# Function to save model checkpoints
def save_checkpoint(config, model, optimizer, epoch, val_loss, best_loss, checkpoint_dir):
    if val_loss < best_loss:
        logger.info(f"Validation loss improved from {best_loss:.2f} to {val_loss:.2f}. Saving model checkpoint.")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(checkpoint_dir, f'best_model_{config["model"]["model_type"]}_{config["year"]}-{config["doy"]}.pth'))
        return val_loss
    return best_loss

def train(config, model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs).squeeze(-1)  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss

        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

        running_loss += loss.item()

        if i >= 1 and config["training"]["overfit_single_batch"]:
            break
    
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    mseloss = nn.MSELoss()
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            if outputs.dim() > 1:  # Check if outputs contain VTEC and uncertainty
                vtec_outputs = outputs[:, 0].squeeze(-1)  # Use only the VTEC part
            else:
                vtec_outputs = outputs.squeeze(-1)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, targets)
            mse = mseloss(vtec_outputs, targets)
            
            running_loss += loss.item()
            running_mse += mse.item()
    
    return running_loss / len(dataloader), running_mse / len(dataloader)

def test(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    criterion = torch.nn.MSELoss()  # You can also use MAE or other regression loss functions

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            if outputs.dim() > 1:
                outputs = outputs[:, 0]
                
            outputs = outputs.squeeze(-1)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)  # Sum loss over the batch
            total_samples += targets.size(0)
    
    mean_loss = total_loss / total_samples
    return mean_loss

def main():
    config = parse_config()
    logger.info(f"Starting training for project: {config['project_name']}")

    setup_seed(config['random_seed'])

    if not config["debugging"]["debug"]:
        wandbname = f"Fusion {config['model']['model_type']} {config['year']}-{config['doy']}"
        wandb.init(project=config['project_name'], name=wandbname, config=config)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    config['device'] = device

    # Initialize dataloaders for train, validation, and test sets
    train_loader, val_loader, test_loader = get_data_loaders(config)

    for x, y in train_loader:
        logger.info(f"Shape of x: {x.shape}, Shape of y: {y.shape}")
        logger.info(f"x: {x[0]}, y: {y[0]}")
        break

    # Initialize model, criterion, optimizer
    model = get_model(config).to(device)
    init_xavier(model, activation=config["model"]["activation"])
    criterion = get_criterion(config)
    optimizer = get_optimizer(config, model.parameters())
    if config["training"]["scheduler"] != None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=config["training"]["scheduler_gamma"])

    # Early stopping and checkpoint setup
    early_stopping_patience = config["training"]["patience"]
    checkpoint_dir = config["logging"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0

    # Training and validation loop
    for epoch in range(config["training"]["epochs"]):
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        train_loss = train(config, model, train_loader, criterion, optimizer, device)
        if not config["training"]["overfit_single_batch"]:
            val_loss, mseloss = validate(model, val_loader, criterion, device)
        else:
            val_loss = best_val_loss
            mseloss = best_val_loss

        if config["training"]["scheduler"] != None: 
            scheduler.step()  # Adjust learning rate

        # Log metrics to wandb
        if not config["debugging"]["debug"]:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_MSE': mseloss,
                'learning_rate': scheduler.get_last_lr()[0],
                'epoch': epoch+1
            })

        logger.info(f"Train Loss: {train_loss:.2f}, Validation Loss: {val_loss:.2f}")
        logger.info(f"Validation MSE: {mseloss:.2f}")

        if val_loss > best_val_loss:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered. Stopping training.")
                break
        else:
            patience_counter = 0

        # Save checkpoint and early stopping logic
        best_val_loss = save_checkpoint(config, model, optimizer, epoch, val_loss, best_val_loss, checkpoint_dir)

    # Final test accuracy
    test_accuracy = test(model, test_loader, device)
    if not config["debugging"]["debug"]:
        wandb.log({'test_MSE': test_accuracy})
    logger.info(f'Test MSE: {test_accuracy:.2f}')

    if not config["debugging"]["debug"]:
        wandb.finish() 

if __name__ == "__main__":
    main()

