import torch
import torch.nn as nn
    
class WeightedLoss(nn.Module):
    def __init__(self, base_loss, mode="GNSS", weighted_loss=False, gnss_weight=1.0, vlbi_weight=1.0):
        super(WeightedLoss, self).__init__()
        self.base_loss = base_loss
        self.mode = mode
        self.weighted_loss = weighted_loss
        self.gnss_weight = gnss_weight
        self.vlbi_weight = vlbi_weight

    def forward(self, outputs, y, technique):
        # Compute per-sample base losses
        base_loss_value = self.base_loss(outputs, y)

        # Return unweighted mean loss if weighting is disabled
        if not self.weighted_loss:
            return base_loss_value.mean()

        # Assign weights based on the technique
        weights = torch.ones_like(base_loss_value)
        if (self.mode == "Fusion" or self.mode =="DTEC_Fusion") and self.weighted_loss:
            weights = torch.where(technique == 0, self.gnss_weight, self.vlbi_weight)

        # Apply weights to per-sample losses
        weighted_loss = torch.mean(weights * base_loss_value)
        return weighted_loss

class LaplaceLoss(nn.Module):
    def __init__(self):
        super(LaplaceLoss, self).__init__()
    
    def forward(self, outputs, y):
        mu, std = outputs[:, 0], outputs[:, 1]
        std = std.reshape(-1,) + 1e-6 
        loss = torch.log(2 * std) + torch.abs(y - mu) / std
        return loss 

class GaussianNLLLoss(nn.Module):
    def __init__(self):
        super(GaussianNLLLoss, self).__init__()
    
    def forward(self, outputs, y):
        # Separate predicted mean and uncertainty (sigma)
        mu, log_sigma = outputs[:, 0], outputs[:, 1]
        sigma = torch.exp(log_sigma) + 1e-6  # Ensure sigma is positive with a small constant for stability
        loss = 0.5 * (((y - mu) ** 2) / (sigma ** 2) + 2 * log_sigma)
        return loss 
    
def get_criterion(config):
    mode = config["data"]["mode"]
    weighted_loss = config["training"]["weighted_loss"]
    loss_type = config["training"]["loss_function"]
    gnss_weight = config["training"]["gnss_loss_weight"]
    vlbi_weight = config["training"]["vlbi_loss_weight"]

    # Initialize the base loss function
    if loss_type == 'MSELoss':
        base_loss = nn.MSELoss(reduction='none')  # Set reduction to 'none' to apply weights later
    elif loss_type == 'MAELoss':
        base_loss = nn.L1Loss(reduction='none')  # Mean Absolute Error
    elif loss_type == 'HuberLoss':
        base_loss = nn.SmoothL1Loss(reduction='none')  # Huber Loss
    elif loss_type == 'LaplaceLoss':
        base_loss = LaplaceLoss()
    elif loss_type == 'GaussianNLLLoss':
        base_loss = GaussianNLLLoss()
    else:
        raise Exception(f'unknown loss {loss_type}')

    # Wrap the base loss function with the WeightedLoss class
    criterion = WeightedLoss(base_loss, mode, weighted_loss, gnss_weight, vlbi_weight)
    return criterion
