"""
Author: Arno Rüegg
Date: 2024-10-21
Description: Global Ionospheric Maps from GNSS and VLBI data
"""

import torch
import torch.optim as optim
import torch.nn as nn
import logging
import wandb
import os

from models.model import get_model
from utils.loss_function import get_criterion
from utils.optimizers import get_optimizer
from utils.config_parser import parse_config
from utils.data import get_data_loaders

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Function to save model checkpoints
def save_checkpoint(config, model, optimizer, epoch, val_loss, best_loss, checkpoint_dir):
    if val_loss < best_loss:
        logger.info(f"Validation loss improved from {best_loss:.4f} to {val_loss:.4f}. Saving model checkpoint.")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(checkpoint_dir, f'best_model_{config["model"]["model_type"]}_{config["year"]}-{config["doy"]}.pth'))
        return val_loss
    return best_loss

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss

        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
    
    return running_loss / len(dataloader)

def test(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    return accuracy

def main():
    config = parse_config()
    logger.info(f"Starting training for project: {config['project_name']}")

    if not config["debugging"]["enable_debugging"]:
        wandbname = f"Fusion {config['model']['model_type']} {config['year']}-{config['doy']}"
        wandb.init(project=config['project_name'], name=wandbname, config=config)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    # Initialize dataloaders for train, validation, and test sets
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # Initialize model, criterion, optimizer
    model = get_model(config).to(device)
    criterion = get_criterion(config)
    optimizer = get_optimizer(config, model.parameters())
    if config["training"]["scheduler"] != None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=config["training"]["scheduler_gamma"])

    # Early stopping and checkpoint setup
    early_stopping_patience = config["training"]["early_stopping_patience"]
    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0

    # Training and validation loop
    for epoch in range(config["training"]["num_epochs"]):
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        if config["training"]["scheduler"] != None: 
            scheduler.step()  # Adjust learning rate

        # Log metrics to wandb
        if not config["debugging"]["enable_debugging"]:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0],
                'epoch': epoch+1
            })

        logger.info(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save checkpoint and early stopping logic
        best_val_loss = save_checkpoint(config, model, optimizer, epoch, val_loss, best_val_loss, checkpoint_dir)

        if val_loss >= best_val_loss:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered. Stopping training.")
                break
        else:
            patience_counter = 0

    # Final test accuracy
    test_accuracy = test(model, test_loader, device)
    if not config["debugging"]["enable_debugging"]:
        wandb.log({'test_accuracy': test_accuracy})
    logger.info(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    if not config["debugging"]["enable_debugging"]:
        wandb.finish() 

if __name__ == "__main__":
    main()

