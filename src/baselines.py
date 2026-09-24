"""PyTorch versions of the four main paper baselines.

Reference: legacy/rnn_fre20-20.ipynb and the saved Keras model configurations.
Each model takes (batch, time, 2) and returns (batch, 1), using a fresh zero
recurrent state for each window. No dropout or output activation is used.
"""

import torch
from torch import nn


def _initialize(model):
    """Match Keras initializer families and its single RNN/LSTM bias."""
    for layer in model.modules():
        if isinstance(layer, (nn.RNN, nn.GRU, nn.LSTM)):
            for name, parameter in layer.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(parameter)
                else:
                    nn.init.zeros_(parameter)
                    if isinstance(layer, (nn.RNN, nn.LSTM)) and "bias_hh" in name:
                        # Keras has only one bias vector in these cells. Keep
                        # PyTorch's redundant second bias fixed to avoid twice
                        # the bias update and extra trainable parameters.
                        parameter.requires_grad_(False)
                    if isinstance(layer, nn.LSTM) and "bias_ih" in name:
                        with torch.no_grad():
                            parameter[layer.hidden_size:2 * layer.hidden_size].fill_(1)
        elif isinstance(layer, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class RNN(nn.Module):
    """Two 32-unit tanh RNN layers followed by a linear force output."""

    def __init__(self):
        super().__init__()
        self.recurrent = nn.RNN(2, 32, num_layers=2, batch_first=True)
        self.output = nn.Linear(32, 1)
        _initialize(self)

    def forward(self, x):
        sequence, _ = self.recurrent(x)
        return self.output(sequence[:, -1])


class LSTM(nn.Module):
    """Two 32-unit LSTM layers with the original unit forget bias."""

    def __init__(self):
        super().__init__()
        self.recurrent = nn.LSTM(2, 32, num_layers=2, batch_first=True)
        self.output = nn.Linear(32, 1)
        _initialize(self)

    def forward(self, x):
        sequence, _ = self.recurrent(x)
        return self.output(sequence[:, -1])


class GRU(nn.Module):
    """Two 32-unit GRU layers, matching Keras reset_after=True."""

    def __init__(self):
        super().__init__()
        self.recurrent = nn.GRU(2, 32, num_layers=2, batch_first=True)
        self.output = nn.Linear(32, 1)
        _initialize(self)

    def forward(self, x):
        sequence, _ = self.recurrent(x)
        return self.output(sequence[:, -1])


class CNNGRU(nn.Module):
    """64-channel width-1 ReLU convolution, two 32-unit GRUs, linear output."""

    def __init__(self):
        super().__init__()
        self.convolution = nn.Conv1d(2, 64, kernel_size=1)
        self.recurrent = nn.GRU(64, 32, num_layers=2, batch_first=True)
        self.output = nn.Linear(32, 1)
        _initialize(self)

    def forward(self, x):
        x = torch.relu(self.convolution(x.transpose(1, 2))).transpose(1, 2)
        sequence, _ = self.recurrent(x)
        return self.output(sequence[:, -1])


class DirectReadout(nn.Module):
    """PGND's encoder and direct readout, without an ODE or history processing.

    This new ablation uses only the last available acceleration/uplift pair.
    It is not a baseline recovered from the previous paper. PyTorch default
    initialization matches PGND's MLP family; the final output starts at zero.
    """

    def __init__(self, encoding_dim=16, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, encoding_dim))
        self.direct_readout = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.direct_readout[-1].weight)
        nn.init.zeros_(self.direct_readout[-1].bias)

    def forward(self, x):
        return self.direct_readout(self.encoder(x[:, -1]))
