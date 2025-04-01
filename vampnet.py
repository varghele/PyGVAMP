import torch
import torch.nn as nn
from typing import Optional


class VAMPNet(nn.Module):
    def __init__(self, encoder, vamp_score, embedding_module=None, lag_time=1):
        """
        Initialize the VAMPNet with a custom encoder and external VAMP score module.

        Args:
            encoder: The encoder network that transforms the input data
            vamp_score: The VAMPScore module for scoring embeddings
            embedding_module: Optional module to create embeddings before the encoder
            lag_time: Lag time for time-lagged datasets
        """
        super(VAMPNet, self).__init__()

        self.encoder = encoder
        self.embedding_module = embedding_module
        self.vamp_score = vamp_score
        self.lag_time = lag_time

    def forward(self, data):
        """
        Transform input data using the embedding module (if provided) and encoder network.

        Args:
            data: Input data (can be graph data or tensor)

        Returns:
            Transformed features
        """
        # Check if data is a PyTorch Geometric Data object
        if hasattr(data, 'x') and hasattr(data, 'edge_index'):
            # Extract graph components
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
            batch = data.batch if hasattr(data, 'batch') else None

            # Apply embedding module to node and edge features if provided
            if self.embedding_module is not None:
                if hasattr(self.embedding_module, 'node_embedding') and hasattr(self.embedding_module,
                                                                                'edge_embedding'):
                    # Module has separate functions for node and edge embeddings
                    x = self.embedding_module.node_embedding(x)
                    if edge_attr is not None and self.embedding_module.edge_embedding is not None:
                        edge_attr = self.embedding_module.edge_embedding(edge_attr)
                else:
                    # Assume single embedding function for nodes
                    x = self.embedding_module(x)

            # Reconstruct PyG data object with embedded features
            embedded_data = type(data)(x=x, edge_index=edge_index)
            if edge_attr is not None:
                embedded_data.edge_attr = edge_attr
            if batch is not None:
                embedded_data.batch = batch

            # Pass through encoder
            return self.encoder(embedded_data)
        else:
            # For non-graph tensor data
            if self.embedding_module is not None:
                # Apply embedding then encoder
                return self.encoder(self.embedding_module(data))
            else:
                # Apply encoder directly
                return self.encoder(data)

    def get_embeddings(self, data):
        """
        Get just the embeddings without passing through the encoder.

        Args:
            data: Input data

        Returns:
            Embedded features
        """
        if self.embedding_module is None:
            return data

        if hasattr(data, 'x') and hasattr(data, 'edge_index'):
            # Graph data
            x = data.x
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

            if hasattr(self.embedding_module, 'node_embedding') and hasattr(self.embedding_module, 'edge_embedding'):
                node_emb = self.embedding_module.node_embedding(x)
                edge_emb = None
                if edge_attr is not None and self.embedding_module.edge_embedding is not None:
                    edge_emb = self.embedding_module.edge_embedding(edge_attr)
                return node_emb, edge_emb
            else:
                return self.embedding_module(x)
        else:
            # Regular tensor data
            return self.embedding_module(data)

    def create_time_lagged_dataset(self, data, lag_time=None):
        """
        Create a time-lagged dataset from sequential data.

        Args:
            data: Sequential data
            lag_time: Lag time to use (defaults to self.lag_time)

        Returns:
            Tuple of (x_t0, x_t1) representing time-lagged pairs
        """
        if lag_time is None:
            lag_time = self.lag_time

        n_samples = len(data) - lag_time
        x_t0 = data[:n_samples]
        x_t1 = data[lag_time:lag_time + n_samples]

        return x_t0, x_t1

    def fit(self, data_loader, optimizer, n_epochs=100, k=None, verbose=True):
        """
        Train the VAMPNet model using the external VAMPScore module.

        Args:
            data_loader: DataLoader providing batches of (x_t0, x_t1)
            optimizer: PyTorch optimizer
            n_epochs: Number of training epochs
            k: Number of singular values to consider for VAMP score
            verbose: Whether to print progress

        Returns:
            List of loss values during training
        """
        losses = []

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in data_loader:
                x_t0, x_t1 = batch

                # Get embeddings
                chi_t0 = self(x_t0)  # Forward pass for time t
                chi_t1 = self(x_t1)  # Forward pass for time t+lag

                # Zero gradients
                optimizer.zero_grad()

                # Calculate VAMP loss using the external VAMPScore module
                loss = self.vamp_score.loss(chi_t0, chi_t1, k=k)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_epoch_loss = epoch_loss / n_batches
            losses.append(avg_epoch_loss)

            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_epoch_loss:.4f}")

        return losses
