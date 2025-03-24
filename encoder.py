# MLP Encoder example
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLPEncoder, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# GNN Encoder example using PyTorch Geometric
class GNNEncoder(nn.Module):
    def __init__(self, node_dim, hidden_dims, output_dim):
        super(GNNEncoder, self).__init__()

        from torch_geometric.nn import GCNConv, global_mean_pool

        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Initial convolution
        self.conv_layers.append(GCNConv(node_dim, hidden_dims[0]))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dims[0]))

        # Hidden convolutions
        for i in range(len(hidden_dims) - 1):
            self.conv_layers.append(GCNConv(hidden_dims[i], hidden_dims[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))

        # Output projection
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, data):
        # Extract node features and edge indices
        x, edge_index = data.x, data.edge_index

        # Graph convolutions
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)

        # Graph pooling
        if hasattr(data, 'batch'):
            x = global_mean_pool(x, data.batch)

        # Project to output dimension
        x = self.output_layer(x)

        return x


# Convolutional Encoder for sequence/image data
class CNNEncoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_dim):
        super(CNNEncoder, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, hidden_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels[0])

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(len(hidden_channels) - 1):
            self.conv_layers.append(
                nn.Conv1d(hidden_channels[i], hidden_channels[i + 1], kernel_size=3, padding=1)
            )
            self.bn_layers.append(nn.BatchNorm1d(hidden_channels[i + 1]))

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels[-1], output_dim)

    def forward(self, x):
        # Assume x has shape [batch_size, input_channels, sequence_length]
        x = torch.relu(self.bn1(self.conv1(x)))

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = torch.relu(bn(conv(x)))

        x = self.pool(x).squeeze(-1)
        x = self.fc(x)

        return x
