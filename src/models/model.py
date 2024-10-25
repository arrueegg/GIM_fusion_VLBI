import torch
import torch.nn as nn
from prettytable import PrettyTable

def get_activation_fn(activation):
    if activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "softmax":
        return nn.Softmax(dim=1)  # Softmax requires specifying a dimension
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "selu":
        return nn.SELU()
    elif activation == "swish":
        return nn.SiLU()  # SiLU is also known as Swish
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

# Initialization function
def init_xavier(model, activation):
    def init_weights(m):
        if isinstance(m, nn.Linear) and m.weight.requires_grad:
            gain = nn.init.calculate_gain(activation)
            torch.nn.init.xavier_normal_(m.weight, gain=gain)
            #torch.nn.init.xavier_uniform_(m.weight, gain=gain)  # Alternative (common choice for small nets)
            if m.bias is not None:
                m.bias.data.fill_(0)
    model.apply(init_weights)

class MLPModel(nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_size, 
                 activation='tanh', apply_softplus=False, dropout_prob=0.2):
        super(MLPModel, self).__init__()
        self.apply_softplus = apply_softplus
        self.dropout_prob = dropout_prob
        self.activation_fn = get_activation_fn(activation)  # Generalized activation function

        # Define layers using nn.Sequential
        layers = []
        layers.append(nn.Linear(input_dimension, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
        layers.append(nn.Linear(hidden_size[-1], output_dimension))
        
        self.layers = nn.Sequential(*layers)

        # Optional Softplus for positive-only outputs
        if apply_softplus:
            self.softplus = nn.Softplus()

        # Define dropout layer (only used in training)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
            if self.training:  # Apply dropout only during training
                x = self.dropout(x)

        x = self.layers[-1](x)  # Output layer

        if self.apply_softplus:
            x = self.softplus(x)
        
        return x

# RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size).to(x.device)  # Initial hidden state
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out

# Model selection function
def get_model(config):
    model_type = config['model']['model_type']
    if model_type == 'MLP':
        return MLPModel(config['model']['input_size'], config['model']['output_size'], 
                        config['model']['hidden_size'], activation=config['model']['activation'], 
                        apply_softplus=config['model']['apply_softplus'], 
                        dropout_prob=config['model']['dropout'])
    elif model_type == 'RNN':
        return RNNModel(config['input_size'], config['hidden_size'], config['output_size'])
    else:
        raise ValueError(f"Model type {model_type} is not recognized. Please select from ['MLP', 'RNN']")

# Function to count the parameters (optimized)
def count_parameters(model):
    with torch.no_grad():  # Disable gradient computation
        table = PrettyTable(['Module Name', 'Parameters'])
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                table.add_row([name, num_params])
                total_params += num_params
        print(table)
        print(f"Total trainable parameters: {total_params}")
    return total_params
