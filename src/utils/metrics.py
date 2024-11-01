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

def calculate_metrics(predictions, targets, techs, prefix):
    """Calculates and returns a dictionary of metrics for each technology type."""

    techs = techs.squeeze(-1)
    metrics = {}

    for tech in [0, 1]:  # Iterate over GNSS (0) and VLBI (1)
        tech_name = "GNSS" if tech == 0 else "VLBI"
        tech_mask = techs == tech
        tech_predictions = predictions[tech_mask]
        tech_targets = targets[tech_mask]

        if tech_predictions.numel() > 0:  # Check if there are any samples for this tech
            tech_prefix = f"{prefix}_{tech_name}"
            metrics.update({
                f'{tech_prefix}_MSE': mse(tech_predictions[:, 0], tech_targets),
                f'{tech_prefix}_MAE': mae(tech_predictions[:, 0], tech_targets),
                f'{tech_prefix}_RMSE': rmse(tech_predictions[:, 0], tech_targets),
                f'{tech_prefix}_MAPE': mape(tech_predictions[:, 0], tech_targets),
                f'{tech_prefix}_R2': r2_score(tech_predictions[:, 0], tech_targets)
            })

            if predictions.shape[1] > 1:  # If there is an uncertainty column
                uncertainty = tech_predictions[:, 1]
                metrics.update({
                    f'{tech_prefix}_uncertainty_mean': uncertainty.mean().item(),
                    f'{tech_prefix}_uncertainty_std': uncertainty.std().item(),
                    f'{tech_prefix}_uncertainty_median': uncertainty.median().item()
                })

    return metrics

