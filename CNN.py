import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, n_input_channels: int, n_hidden_layers: int, n_hidden_kernels: int, kernel_size: int, activation_fn: torch.autograd.Function):
        """CNN, consisting of "n_hidden_layers" linear layers, using relu
        activation function in the hidden CNN layers.

        Parameters
        ----------
        n_input_channels: int
            Number of features channels in input tensor
        n_hidden_layers: int
            Number of hidden layers
        n_hidden_kernels: int
            Number of kernels in each hidden layer
        n_output_channels: int
            Number of features in output tensor
        """
        super().__init__()

        hidden_layers = []
        for i in range(n_hidden_layers):
            layer = nn.Conv2d(in_channels=n_input_channels, out_channels=n_hidden_kernels, kernel_size=kernel_size,bias=True, padding=int(kernel_size/2))
            hidden_layers.append(layer)
            # Add relu activation module to list of modules
            hidden_layers.append(activation_fn)
            hidden_layers.append(nn.Dropout(0.2))
            n_input_channels = n_hidden_kernels

        self.hidden_layers = nn.Sequential(*hidden_layers)
        #Todo: change output layer, isn't correct yet
        self.output_layer = nn.Conv2d(in_channels=n_input_channels, out_channels=3, kernel_size=kernel_size,bias=True, padding=int(kernel_size/2))

    def forward(self, x: torch.Tensor):
        """Apply CNN to "x"

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (n_samples, n_input_channels, x, y)

        Returns
        ----------
        torch.Tensor
            Output tensor of shape (n_samples, n_output_channels, u, v)
        """
        # Apply hidden layers module
        hidden_features = self.hidden_layers(x)

        # Apply last layer (=output layer)
        output = self.output_layer(hidden_features)
        return output