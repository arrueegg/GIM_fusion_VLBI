import torch
import torch.nn as nn

# Define your MLP model
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Another example model, a simple RNN model
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
        return MLPModel(config['input_size'], config['hidden_size'], config['output_size'])
    elif model_type == 'RNN':
        return RNNModel(config['input_size'], config['hidden_size'], config['output_size'])
    else:
        raise ValueError(f"Model type {model_type} is not recognized. Please select from ['MLP', 'RNN']")
