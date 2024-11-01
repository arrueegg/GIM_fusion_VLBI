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
from utils.metrics import calculate_metrics

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
    logger.info(f"Validation loss improved from {best_loss:.2f} to {val_loss:.2f}. Saving model checkpoint.")
    torch.save({
        'epoch': epoch,
        'model_type': config['model']['model_type'],
        'input_size': config['model']['input_size'],
        'output_size': config['model']['output_size'],
        'hidden_size': config['model']['hidden_size'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, os.path.join(checkpoint_dir, f'best_model_{config["data"]["mode"]}_{config["model"]["model_type"]}_{config["year"]}-{config["doy"]}.pth'))
    return val_loss

def train(config, model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    all_outputs = []
    all_targets = []
    techs = []

    for i, (inputs, targets, tech) in tqdm(enumerate(dataloader), total=len(dataloader)):

        inputs, targets, tech = inputs.to(device), targets.to(device), tech.to(device)

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs).squeeze(-1)  # Forward pass
        loss = criterion(outputs, targets, tech)  # Compute loss

        #calculate_metrics(outputs, targets, tech, prefix="train")

        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

        running_loss += loss.item()
        all_outputs.append(outputs.detach().cpu())
        all_targets.append(targets.detach().cpu())
        techs.append(tech.detach().cpu())

        if i >= 1 and config["training"]["overfit_single_batch"]:
            break
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss, torch.cat(all_outputs), torch.cat(all_targets), torch.cat(techs)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_targets = []
    techs = []
    
    with torch.no_grad():
        for inputs, targets, tech in dataloader:
            inputs, targets, tech = inputs.to(device), targets.to(device), tech.to(device)
            
            outputs = model(inputs)
            if outputs.dim() > 1:  # Check if outputs contain VTEC and uncertainty
                vtec_outputs = outputs[:, 0].squeeze(-1)  # Use only the VTEC part
            else:
                vtec_outputs = outputs.squeeze(-1)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, targets, tech)
            
            running_loss += loss.item()
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
            techs.append(tech.cpu())
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss, torch.cat(all_outputs), torch.cat(all_targets), torch.cat(techs)

def test(model, dataloader, device):
    model.eval()
    all_outputs = []
    all_targets = []
    techs = []

    with torch.no_grad():
        for inputs, targets, tech in dataloader:
            inputs, targets, tech = inputs.to(device), targets.to(device), tech.to(device)
            
            outputs = model(inputs)
            if outputs.dim() > 1:
                outputs = outputs[:, 0]
                
            outputs = outputs.squeeze(-1)
            
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
            techs.append(tech.cpu())
    
    return torch.cat(all_outputs), torch.cat(all_targets), torch.cat(techs)

def main():
    config = parse_config()
    logger.info(f"Starting training for project: {config['project_name']}")

    setup_seed(config['random_seed'])

    if not config["debugging"]["debug"]:
        wandbname = f"{config['data']['mode']} {config['model']['model_type']} {config['year']}-{config['doy']}"
        wandb.init(project=config['project_name'], name=wandbname, config=config)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    config['device'] = device

    # Initialize dataloaders for train, validation, and test sets
    train_loader, val_loader, test_loader = get_data_loaders(config)

    for x, y, tech in train_loader:
        logger.info(f"Trainloader:      Shape of x: {x.shape}, Shape of y: {y.shape}, Shape of tech: {tech.shape}")
        #logger.info(f"x: {x[0]}, y: {y[0]}, tech: {tech[0]}")
        break

    for x, y, tech in val_loader:
        logger.info(f"Validationloader: Shape of x: {x.shape}, Shape of y: {y.shape}, Shape of tech: {tech.shape}")
        #logger.info(f"x: {x[0]}, y: {y[0]}, tech: {tech[0]}")
        break

    for x, y, tech in test_loader:
        logger.info(f"Testloader:       Shape of x: {x.shape}, Shape of y: {y.shape}, Shape of tech: {tech.shape}")
        #logger.info(f"x: {x[0]}, y: {y[0]}, tech: {tech[0]}")
        break

    # Initialize model, criterion, optimizer
    model = get_model(config).to(device)
    init_xavier(model, activation=config["model"]["activation"])
    criterion = get_criterion(config)
    optimizer = get_optimizer(config, model.parameters())
    if config["training"]["scheduler"] != None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"],
                                                     gamma=config["training"]["scheduler_gamma"])

    # Early stopping and checkpoint setup
    checkpoint_dir = config["logging"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0

    # Training and validation loop
    for epoch in range(config["training"]["epochs"]):
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        train_loss, train_outputs, train_targets, train_techs = train(config, model, train_loader, criterion, optimizer, device)
        if not config["training"]["overfit_single_batch"]:
            val_loss, val_outputs, val_targets, val_techs = validate(model, val_loader, criterion, device)
        else:
            val_loss = best_val_loss

        # Calculate and log metrics
        if not config["debugging"]["debug"]:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0],
                **calculate_metrics(train_outputs, train_targets, techs=train_techs, prefix="train"),
                **calculate_metrics(val_outputs, val_targets, techs=val_techs, prefix="val"),
                'epoch': epoch + 1
            })

        if config["training"]["scheduler"] != None: 
            scheduler.step()  # Adjust learning rate

        logger.info(f"Train Loss: {train_loss:.2f}, Validation Loss: {val_loss:.2f}")

        if val_loss < best_val_loss:
            patience_counter = 0
            best_val_loss = save_checkpoint(config, model, optimizer, epoch, val_loss, best_val_loss, checkpoint_dir)
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["patience"]:
                logger.info("Early stopping triggered. Stopping training.")
                break

    # Load best model for final testing
    checkpoint = torch.load(os.path.join(checkpoint_dir, f'best_model_{config["data"]["mode"]}_{config["model"]["model_type"]}_{config["year"]}-{config["doy"]}.pth'), weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final test accuracy
    test_outputs, test_targets, test_techs = test(model, test_loader, device)
    test_metrics = calculate_metrics(test_outputs, test_targets, techs=test_techs, prefix="test")
    if not config["debugging"]["debug"]:
        wandb.log(test_metrics)
    formatted_test_metrics = {k: f"{v:.2f}" for k, v in test_metrics.items()}
    logger.info(f'Test metrics: {formatted_test_metrics}')

    if not config["debugging"]["debug"]:
        wandb.finish() 

if __name__ == "__main__":
    main()

