import torch
import torch.functional as F


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, activation, batchnorm=False, layernorm=False, dropout=0.0):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if activation == 'relu':
            self.activation = torch.nn.ReLU
        else:
            raise NotImplementedError
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.dropout = dropout
        assert not self.batchnorm
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(self.input_size, self.hidden_size))
        if self.batchnorm:
            self.layers.append(torch.nn.BatchNorm1d(self.hidden_size))
        if self.layernorm:
            self.layers.append(torch.nn.LayerNorm(self.hidden_size))
        self.layers.append(self.activation())
        if self.dropout > 0.0:
            self.layers.append(torch.nn.Dropout(self.dropout))
        for _ in range(self.num_layers - 1):
            self.layers.append(torch.nn.Linear(self.hidden_size, self.hidden_size))
            if self.batchnorm:
                self.layers.append(torch.nn.BatchNorm1d(self.hidden_size))
            if self.layernorm:
                self.layers.append(torch.nn.LayerNorm(self.hidden_size))
            self.layers.append(self.activation())
            if self.dropout > 0.0:
                self.layers.append(torch.nn.Dropout(self.dropout))
        self.layers.append(torch.nn.Linear(self.hidden_size, self.output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
