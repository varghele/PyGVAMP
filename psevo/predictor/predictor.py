import torch.nn as nn

class Predictor(nn.Module):
    """ Molecular Property Predictor using 2-layer MLP.
    Predicts molecular properties from graph-level embeddings.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): Dimension of input graph-level features
            hidden_dim (int): Dimension of hidden layers
            output_dim (int): Number of molecular properties to predict
        """
        super(Predictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 2-layer MLP with ReLU activation
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Output layer for property regression
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): Graph-level embeddings [batch_size, input_dim]

        Returns:
            Tensor: Predicted molecular properties [batch_size, output_dim]
        """
        hidden_features = self.mlp(x)
        predicted_properties = self.output_layer(hidden_features)
        return predicted_properties

    def get_model_info(self):
        """Get model configuration and parameter count."""
        num_params = sum(p.numel() for p in self.parameters())

        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_parameters': num_params,
            'architecture': '2-layer MLP'
        }
