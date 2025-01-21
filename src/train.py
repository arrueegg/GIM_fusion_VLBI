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
from utils.data_SH import get_data_loaders, get_stations
from utils.metrics import calculate_metrics
from utils.mapping_function import MappingFunction as MF

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
def save_checkpoint(config, model, optimizer, epoch, val_loss, best_loss, checkpoint_dir, model_seed):
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
    }, os.path.join(checkpoint_dir, f'best_model_{config["data"]["mode"]}_{config["model"]["model_type"]}_{config["year"]}-{config["doy"]}_seed{model_seed:02}.pth'))
    return val_loss

def train(config, model, vlbi_offsets, dataloader, criterion, optimizer, mf, device):
    model.train()
    running_loss = 0.0

    all_outputs = []
    all_targets = []
    techs = []

    for i, (inputs, targets, tech) in tqdm(enumerate(dataloader), total=len(dataloader), disable=config["debugging"]["disable_tqdm"]):

        inputs, targets, tech = inputs.to(device), targets.to(device), tech.to(device)
        if config["data"]["mode"] == "DTEC_Fusion":
            inputs, sta_ind, elev = inputs[:, :-2], inputs[:, -2], inputs[:, -1]  # Separate elevation data

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs).squeeze(-1)  # Forward pass

        # Separate GNSS and VLBI data
        gnss_mask = tech == 0
        vlbi_mask = tech == 1

        gnss_outputs = outputs[gnss_mask]
        gnss_targets = targets[gnss_mask]
        gnss_tech = tech[gnss_mask]

        vlbi_outputs = outputs[vlbi_mask]
        vlbi_targets = targets[vlbi_mask]
        vlbi_tech = tech[vlbi_mask]
        if config["data"]["mode"] == "DTEC_Fusion":
            vlbi_elev = elev[vlbi_mask]
            vlbi_sta_ind = sta_ind[vlbi_mask]

        # For VLBI data, subtract subsequent predictions
        if vlbi_outputs.size(0) > 0 and config["data"]["mode"] == "DTEC_Fusion":
            #instr_offsets_adjusted = vlbi_offsets - vlbi_offsets.mean()
            instr_offset1, instr_offset2 = vlbi_offsets[vlbi_sta_ind[0::2].long()], vlbi_offsets[vlbi_sta_ind[1::2].long()]
            mfs = mf(vlbi_elev)
            vlbi_out1_mapped, vlbi_out2_mapped = vlbi_outputs[0::2, 0] * mfs[0::2], vlbi_outputs[1::2, 0] * mfs[1::2]
            vlbi_outputs = torch.stack((vlbi_out1_mapped - vlbi_out2_mapped + instr_offset2 - instr_offset1, (vlbi_outputs[0::2, 1] + vlbi_outputs[1::2, 1]) / 2), dim=1)
            vlbi_targets = vlbi_targets[0::2]
            vlbi_tech = vlbi_tech[0::2]

        # Combine GNSS and VLBI outputs and targets for loss calculation
        combined_outputs = torch.cat([gnss_outputs, vlbi_outputs])
        combined_targets = torch.cat([gnss_targets, vlbi_targets])
        tech = torch.cat([gnss_tech, vlbi_tech])

        loss = criterion(combined_outputs, combined_targets, tech)
        loss += torch.sum(vlbi_offsets) ** 2

        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

        running_loss += loss.item()
        all_outputs.append(combined_outputs.detach().cpu())
        all_targets.append(combined_targets.detach().cpu())
        techs.append(tech.detach().cpu())

        if i >= 1 and config["training"]["overfit_single_batch"]:
            break

    avg_loss = running_loss / len(all_targets)
    return avg_loss, torch.cat(all_outputs), torch.cat(all_targets), torch.cat(techs)

def validate(config, model, vlbi_offsets, dataloader, criterion, mf, device):
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_targets = []
    techs = []
    
    with torch.no_grad():
        for inputs, targets, tech in dataloader:
            inputs, targets, tech = inputs.to(device), targets.to(device), tech.to(device)
            if config["data"]["mode"] == "DTEC_Fusion":
                inputs, sta_ind, elev = inputs[:, :-2], inputs[:, -2], inputs[:, -1]  # Separate elevation data

            outputs = model(inputs).squeeze(-1)
            
            # Separate GNSS and VLBI data
            gnss_mask = tech == 0
            vlbi_mask = tech == 1

            gnss_outputs = outputs[gnss_mask]
            gnss_targets = targets[gnss_mask]
            gnss_tech = tech[gnss_mask]

            vlbi_outputs = outputs[vlbi_mask]
            vlbi_targets = targets[vlbi_mask]
            vlbi_tech = tech[vlbi_mask]
            if config["data"]["mode"] == "DTEC_Fusion":
                vlbi_elev = elev[vlbi_mask]
                vlbi_sta_ind = sta_ind[vlbi_mask]

            # For VLBI data, subtract subsequent predictions
            if vlbi_outputs.size(0) > 0 and config["data"]["mode"] == "DTEC_Fusion":
                instr_offset1, instr_offset2 = vlbi_offsets[vlbi_sta_ind[0::2]], vlbi_offsets[vlbi_sta_ind[1::2]]
                mfs = mf(vlbi_elev)
                vlbi_out1_mapped, vlbi_out2_mapped = vlbi_outputs[0::2, 0] * mfs[0::2], vlbi_outputs[1::2, 0] * mfs[1::2]
                vlbi_outputs = torch.stack((vlbi_out1_mapped - vlbi_out2_mapped + instr_offset2 - instr_offset1, (vlbi_outputs[0::2, 1] + vlbi_outputs[1::2, 1]) / 2), dim=1)
                vlbi_targets = vlbi_targets[0::2]
                vlbi_tech = vlbi_tech[0::2]

            # Combine GNSS and VLBI outputs and targets for loss calculation
            combined_outputs = torch.cat([gnss_outputs, vlbi_outputs])
            combined_targets = torch.cat([gnss_targets, vlbi_targets])
            tech = torch.cat([gnss_tech, vlbi_tech])

            loss = criterion(combined_outputs, combined_targets, tech)
            
            running_loss += loss.item()
            all_outputs.append(combined_outputs.cpu())
            all_targets.append(combined_targets.cpu())
            techs.append(tech.cpu())
    
    avg_loss = running_loss / len(all_targets)
    return avg_loss, torch.cat(all_outputs), torch.cat(all_targets), torch.cat(techs)

def test(config, model, vlbi_offsets, dataloader, mf, device):
    model.eval()
    all_outputs = []
    all_targets = []
    techs = []

    with torch.no_grad():
        for inputs, targets, tech in dataloader:
            inputs, targets, tech = inputs.to(device), targets.to(device), tech.to(device)
            if config["data"]["mode"] == "DTEC_Fusion":
                inputs, sta_ind, elev = inputs[:, :-2], inputs[:, -2], inputs[:, -1]

            outputs = model(inputs).squeeze(-1)
            
            # Separate GNSS and VLBI data
            gnss_mask = tech == 0
            vlbi_mask = tech == 1

            gnss_outputs = outputs[gnss_mask]
            gnss_targets = targets[gnss_mask]
            gnss_tech = tech[gnss_mask]

            vlbi_outputs = outputs[vlbi_mask]
            vlbi_targets = targets[vlbi_mask]
            vlbi_tech = tech[vlbi_mask]
            if config["data"]["mode"] == "DTEC_Fusion":
                vlbi_elev = elev[vlbi_mask]
                vlbi_sta_ind = sta_ind[vlbi_mask]

            # For VLBI data, subtract subsequent predictions
            if vlbi_outputs.size(0) > 0 and config["data"]["mode"] == "DTEC_Fusion":
                instr_offset1, instr_offset2 = vlbi_offsets[vlbi_sta_ind[0::2]], vlbi_offsets[vlbi_sta_ind[1::2]]
                mfs = mf(vlbi_elev)
                vlbi_out1_mapped, vlbi_out2_mapped = vlbi_outputs[0::2, 0] * mfs[0::2], vlbi_outputs[1::2, 0] * mfs[1::2]
                vlbi_outputs = torch.stack((vlbi_out1_mapped - vlbi_out2_mapped + instr_offset2 - instr_offset1, (vlbi_outputs[0::2, 1] + vlbi_outputs[1::2, 1]) / 2), dim=1)
                vlbi_targets = vlbi_targets[0::2]
                vlbi_tech = vlbi_tech[0::2]

            # Combine GNSS and VLBI outputs and targets
            combined_outputs = torch.cat([gnss_outputs, vlbi_outputs])
            combined_targets = torch.cat([gnss_targets, vlbi_targets])
            tech = torch.cat([gnss_tech, vlbi_tech])

            all_outputs.append(combined_outputs.cpu())
            all_targets.append(combined_targets.cpu())
            techs.append(tech.cpu())
    
    return torch.cat(all_outputs), torch.cat(all_targets), torch.cat(techs)

def main():
    logger.info(f"num threads {torch.get_num_threads()}")  # Number of threads PyTorch is using
    logger.info(f"inter-op threads {torch.get_num_interop_threads()}")  # For inter-op parallelism
    config = parse_config()
    logger.info(f"Starting ensemble training with {config['model']['ensemble_size']} member/s")

    setup_seed(config['random_seed'])

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    config['device'] = device

    # Initialize dataloaders for train, validation, and test sets
    train_loader, val_loader, test_loader = get_data_loaders(config)
    if config["data"]["mode"] == "DTEC_Fusion":
        vlbi_stations = get_stations(config)
    else:
        vlbi_stations = []
    
    """for x, y, tech in train_loader:
        logger.info(f"Trainloader:      Shape of x: {x.shape}, Shape of y: {y.shape}, Shape of tech: {tech.shape}")
        logger.info(f"x: {x[0]}, y: {y[0]}")
        logger.info(f"x: {x[-1]}, y: {y[-1]}")
        break
    
    for x, y, tech in val_loader:
        logger.info(f"Validationloader: Shape of x: {x.shape}, Shape of y: {y.shape}, Shape of tech: {tech.shape}")
        logger.info(f"x: {x[0]}, y: {y[0]}")
        logger.info(f"x: {x[-1]}, y: {y[-1]}")
        break

    for x, y, tech in test_loader:
        logger.info(f"Testloader:       Shape of x: {x.shape}, Shape of y: {y.shape}, Shape of tech: {tech.shape}")
        logger.info(f"x: {x[0]}, y: {y[0]}")
        logger.info(f"x: {x[-1]}, y: {y[-1]}")
        break"""

    # Ensemble configuration
    ensemble_size = config['model']['ensemble_size']
    model_dir = os.path.join(config['output_dir'], 'model')
    os.makedirs(model_dir, exist_ok=True)

    # Train ensemble models
    for model_seed in range(ensemble_size):
        logger.info(f"Training ensemble model {model_seed+1}/{ensemble_size}")

        if not config["debugging"]["debug"]:
            wandbname = f"{config['data']['mode']} {config['model']['model_type']} {config['year']}-{config['doy']} m{1+model_seed:02}"
            wandb.init(project=config['project_name'], name=wandbname, config=config, mode="offline")
        
        # Initialize model, criterion, optimizer
        model = get_model(config).to(device)
        init_xavier(model, activation=config["model"]["activation"], model_seed=model_seed)
        instr_offsets = torch.nn.Parameter(torch.zeros(len(vlbi_stations)).to(device), requires_grad=True)
        criterion = get_criterion(config)
        optimizer = get_optimizer(config, list(model.parameters()) + [instr_offsets])
        mf = MF.mslm
        scheduler = config["training"]["scheduler"]
        if scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config["training"]["scheduler_step_size"], 
                gamma=config["training"]["scheduler_gamma"]
            )

        best_val_loss = float('inf')
        patience_counter = 0

        # Training and validation loop
        for epoch in range(config["training"]["epochs"]):
            logger.info(f"Model {model_seed+1}/{ensemble_size}, Epoch {epoch+1}/{config['training']['epochs']}")
            
            train_loss, train_outputs, train_targets, train_techs = train(config, model, instr_offsets, train_loader, criterion, optimizer, mf, device)
            if not config["training"]["overfit_single_batch"]:
                val_loss, val_outputs, val_targets, val_techs = validate(config, model, instr_offsets, val_loader, criterion, mf, device)
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

            if scheduler:
                scheduler.step()

            logger.info(f"Model {model_seed+1}/{ensemble_size}, Train Loss: {train_loss:.2f}, Validation Loss: {val_loss:.2f}")

            if val_loss < best_val_loss or config["training"]["save_model_every_epoch"]:
                patience_counter = 0
                best_val_loss = save_checkpoint(config, model, optimizer, epoch, val_loss, best_val_loss, model_dir, model_seed)
            else:
                patience_counter += 1
                if patience_counter >= config["training"]["patience"]:
                    logger.info(f"Early stopping triggered for model {model_seed+1}/{ensemble_size}.")
                    break

        # Load best model for final testing
        checkpoint = torch.load(os.path.join(model_dir, f'best_model_{config["data"]["mode"]}_{config["model"]["model_type"]}_{config["year"]}-{config["doy"]}_seed{model_seed:02}.pth'), weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

        # print Instrumental Offsets
        logger.info(f"Instr Offsets: {instr_offsets.data}")

        # Final test accuracy
        test_outputs, test_targets, test_techs = test(config, model, instr_offsets, test_loader, mf, device)
        test_metrics = calculate_metrics(test_outputs, test_targets, techs=test_techs, prefix="test")

        formatted_test_metrics = {k: f"{v:.2f}" for k, v in test_metrics.items()}
        logger.info(f'Test metrics: {formatted_test_metrics}')
        if not config["debugging"]["debug"]:
            wandb.log(test_metrics)
            wandb.finish()

    # Ensemble testing
    logger.info("Testing ensemble models...")
    all_predictions = []
    all_targets = []
    all_techs = []
    for model_seed in range(ensemble_size):
        # Load best model for final testing
        checkpoint = torch.load(os.path.join(model_dir, f'best_model_{config["data"]["mode"]}_{config["model"]["model_type"]}_{config["year"]}-{config["doy"]}_seed{model_seed:02}.pth'), weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        outputs, targets, tech = test(config, model, instr_offsets, test_loader, mf, device)
        all_predictions.append(outputs)
        all_targets.append(targets)
        all_techs.append(tech)

    # Combine predictions (e.g., average for regression)
    ensemble_predictions = torch.mean(torch.stack(all_predictions), dim=0)

    # Calculate metrics
    test_metrics = calculate_metrics(ensemble_predictions, all_targets[0], all_techs[0], prefix="test")

    formatted_test_metrics = {k: f'{v:.2f}' for k, v in test_metrics.items()}
    logger.info(f"Ensemble Test Metrics: {formatted_test_metrics}")


if __name__ == "__main__":
    main()

