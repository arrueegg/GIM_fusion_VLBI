import torch
import torch.optim as optim

def get_optimizer(config, model_parameters):
    optimizer_type = config['training']['optimizer']
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]

    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model_parameters, lr=lr, weight_decay=weight_decay)  # Stochastic Gradient Descent
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)  # Adam optimizer
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)  # Adam with weight decay fix
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model_parameters, lr=lr, weight_decay=weight_decay)  # RMSprop optimizer
    elif optimizer_type == 'Adagrad':
        optimizer = optim.Adagrad(model_parameters, lr=lr, weight_decay=weight_decay)  # Adagrad optimizer
    elif optimizer_type == 'Adadelta':
        optimizer = optim.Adadelta(model_parameters, lr=lr, weight_decay=weight_decay)  # Adadelta optimizer
    else:
        raise Exception(f'Unknown optimizer {optimizer_type}')

    return optimizer