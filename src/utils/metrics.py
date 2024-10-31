import torch
import torch.nn.functional as F

def mse(predictions, targets):
    return F.mse_loss(predictions, targets).item()

def mae(predictions, targets):
    return F.l1_loss(predictions, targets).item()

def rmse(predictions, targets):
    return torch.sqrt(F.mse_loss(predictions, targets)).item()

def mape(predictions, targets):
    epsilon = 1e-7  # to avoid division by zero
    return (torch.mean(torch.abs((targets - predictions) / (targets + epsilon))) * 100).item()

def r2_score(predictions, targets):
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    return (1 - ss_res / ss_tot).item()

def calculate_metrics(predictions, targets, prefix=""):
    """Calculates and returns a dictionary of metrics with an optional prefix for logging."""

    if predictions.dim() > 1:  # Check if outputs contain VTEC and uncertainty
        vtec_predictions = predictions[:, 0].squeeze(-1)
        uncertainty = predictions[:, 1].squeeze(-1)
    else:
        vtec_predictions = predictions.squeeze(-1)
        uncertainty = None

    # Main metrics for VTEC predictions
    metrics = {
        f'{prefix}_MSE': mse(vtec_predictions, targets),
        f'{prefix}_MAE': mae(vtec_predictions, targets),
        f'{prefix}_RMSE': rmse(vtec_predictions, targets),
        f'{prefix}_MAPE': mape(vtec_predictions, targets),
        f'{prefix}_R2': r2_score(vtec_predictions, targets),
    }

    # Additional uncertainty metrics if available
    if uncertainty is not None:
        metrics[f'{prefix}_uncertainty_mean'] = uncertainty.mean().item()
        metrics[f'{prefix}_uncertainty_std'] = uncertainty.std().item()
        metrics[f'{prefix}_uncertainty_median'] = uncertainty.median().item()

    return metrics

