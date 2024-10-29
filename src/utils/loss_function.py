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

class GaussianNLLLoss(nn.Module):
    def __init__(self):
        super(GaussianNLLLoss, self).__init__()
    
    def forward(self, outputs, y):
        # Separate predicted mean and uncertainty (sigma)
        mu, log_sigma = outputs[:, 0], outputs[:, 1]
        sigma = torch.exp(log_sigma) + 1e-6  # Ensure sigma is positive with small constant for stability
        
        # Calculate the Gaussian Negative Log Likelihood Loss
        loss = 0.5 * torch.mean(((y - mu) ** 2) / (sigma ** 2) + 2 * log_sigma)
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
    elif loss_type == 'GaussianNLLLoss':
        criterion = GaussianNLLLoss()
    else:
        raise Exception(f'unknown loss {loss_type}')

    return criterion