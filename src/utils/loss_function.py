import torch
import torch.nn as nn

def get_criterion(config):
    loss_type = config["training"]["loss_function"]

    if loss_type == 'MSELoss':
        criterion = nn.MSELoss()  # Mean Squared Error
    elif loss_type == 'MAELoss':
        criterion = nn.L1Loss()  # Mean Absolute Error
    elif loss_type == 'HuberLoss':
        criterion = nn.SmoothL1Loss()  # Huber Loss
    else:
        raise Exception(f'unknown loss {loss_type}')

    return criterion