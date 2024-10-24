import torch
import torch.nn as nn

class LaplaceLoss(nn.Module):
    def __init__(self):
        super(LaplaceLoss, self).__init__()
    
    def forward(self, outputs, y):
        mu, std = outputs[:,0], outputs[:,1]
        std = std.reshape(-1, ) + 1e-6  # Ensure stability with small constant
        loss = torch.mean(torch.log(2 * std) + torch.abs(y - mu) / std)
        return loss
    
def get_criterion(config):
    loss_type = config["training"]["loss_function"]

    if loss_type == 'MSELoss':
        criterion = nn.MSELoss()  # Mean Squared Error
    elif loss_type == 'MAELoss':
        criterion = nn.L1Loss()  # Mean Absolute Error
    elif loss_type == 'HuberLoss':
        criterion = nn.SmoothL1Loss()  # Huber Loss
    elif loss_type == 'LaplaceLoss':
        criterion = LaplaceLoss()
    else:
        raise Exception(f'unknown loss {loss_type}')

    return criterion