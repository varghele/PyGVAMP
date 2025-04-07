import torch
import torch.nn as nn
from torch_geometric.nn.models import MLP


class SoftmaxMLP(nn.Module):
    """
    MLP with softmax activation on the final layer.

    Parameters:
    -----------
    in_channels : int
        Input feature dimension
    hidden_channels : int or list
        Hidden layer dimension(s)
    out_channels : int
        Output dimension (number of classes)
    num_layers : int
        Total number of layers
    dropout : float, optional
        Dropout probability
    act : str or callable, optional
        Activation function for hidden layers
    norm : str or callable, optional
        Normalization function
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0.0, act='relu', norm=None):
        super(SoftmaxMLP, self).__init__()

        # Create MLP for all but the last layer
        if num_layers > 1:
            # If more than one layer, use MLP for hidden layers
            self.mlp = MLP(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,  # Output is hidden dim
                num_layers=num_layers - 1,  # One less layer
                dropout=dropout,
                act=act,
                norm=norm
            )
            # Add final layer with softmax
            self.final_layer = nn.Sequential(
                nn.Linear(hidden_channels, out_channels),
                nn.Softmax(dim=1)
            )
        else:
            # If only one layer, create a simple Linear + Softmax
            self.mlp = None
            self.final_layer = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        if self.mlp is not None:
            x = self.mlp(x)
        return self.final_layer(x)
