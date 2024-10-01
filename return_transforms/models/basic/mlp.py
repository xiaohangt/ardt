import torch


class MLP(torch.nn.Module):
    """
    Multi-layer perceptron (MLP) model.
    
    Args:
        input_size (int): Size of the input layer.
        output_size (int): Size of the output layer.
        hidden_size (int): Number of units in the hidden layers.
        num_layers (int): Number of hidden layers.
        activation (str, optional): Activation function to use ('relu'). Default is 'relu'.
        dropout (float, optional): Dropout rate. Default is 0.0.
        batchnorm (bool, optional): Whether to apply batch normalization. Default is False.
        layernorm (bool, optional): Whether to apply layer normalization. Default is False.
    """
    
    def __init__(
            self, 
            input_size: int, 
            output_size: int, 
            hidden_size: int, 
            num_layers: int, 
            activation: str = 'relu', 
            dropout: float = 0.0,
            batchnorm: bool = False, 
            layernorm: bool = False
        ):
        super(MLP, self).__init__()
        
        # Initialize model parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Set the activation function (only ReLU is currently implemented)
        if activation == 'relu':
            self.activation = torch.nn.ReLU
        else:
            raise NotImplementedError(f'Activation {activation} not implemented')
        
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.dropout = dropout
        
        # Ensure that batchnorm is not used (assertion to safeguard)
        assert not self.batchnorm
        
        # Initialize the list of layers in the MLP
        self.layers = torch.nn.ModuleList()
        
        # Add the first layer (input to hidden)
        self.layers.append(torch.nn.Linear(self.input_size, self.hidden_size))
        
        # Optionally, apply batch normalization after the first layer
        if self.batchnorm:
            self.layers.append(torch.nn.BatchNorm1d(self.hidden_size))
        
        # Optionally, apply layer normalization after the first layer
        if self.layernorm:
            self.layers.append(torch.nn.LayerNorm(self.hidden_size))
        
        # Add activation after the first layer
        self.layers.append(self.activation())
        
        # Optionally, apply dropout after the first layer
        if self.dropout > 0.0:
            self.layers.append(torch.nn.Dropout(self.dropout))
        
        # Add additional hidden layers (repeating structure)
        for _ in range(self.num_layers - 1):
            self.layers.append(torch.nn.Linear(self.hidden_size, self.hidden_size))
            
            # Optionally, apply batch normalization for hidden layers
            if self.batchnorm:
                self.layers.append(torch.nn.BatchNorm1d(self.hidden_size))
            
            # Optionally, apply layer normalization for hidden layers
            if self.layernorm:
                self.layers.append(torch.nn.LayerNorm(self.hidden_size))
            
            # Add activation after each hidden layer
            self.layers.append(self.activation())
            
            # Optionally, apply dropout after each hidden layer
            if self.dropout > 0.0:
                self.layers.append(torch.nn.Dropout(self.dropout))
        
        # Add the final layer (hidden to output)
        self.layers.append(torch.nn.Linear(self.hidden_size, self.output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        # Pass the input through each layer in the network
        for layer in self.layers:
            x = layer(x)
        return x
