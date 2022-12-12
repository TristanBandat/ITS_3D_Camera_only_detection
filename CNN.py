import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, n_input_channels: int, n_hidden_layers: int, n_hidden_kernels: int, kernel_size: int):
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
            layer = nn.Conv2d(in_channels=n_input_channels, out_channels=n_hidden_kernels, kernel_size=kernel_size,
                              bias=False, padding=int(kernel_size / 2))
            hidden_layers.append(layer)
            # Add relu activation module to list of modules
            hidden_layers.append(torch.nn.ReLU())
            hidden_layers.append(nn.Dropout(0.2))
            n_input_channels = n_hidden_kernels

        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.output_layer = nn.Conv2d(in_channels=n_input_channels, out_channels=1, kernel_size=1)

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

        # maps (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y) and in the last layer
        # (N, n_kernels, X, Y) -> (N, 3, X, Y)
        return self.output_layer(self.hidden_layers(x))
