import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from classifier.SoftmaxMLP import SoftmaxMLP
from torch_geometric.nn.models import MLP

class VAMPNet(nn.Module):
    def __init__(self,
                 encoder,
                 vamp_score,
                 # Embedding options
                 embedding_module=None,
                 embedding_in_dim=None,
                 embedding_out_dim=None,
                 embedding_hidden_dim=64,
                 embedding_num_layers=2,
                 embedding_dropout=0.0,
                 embedding_act='relu',
                 embedding_norm=None,
                 # Classifier options
                 classifier_module=None,
                 n_classes=None,
                 classifier_hidden_dim=64,
                 classifier_num_layers=2,
                 classifier_dropout=0.0,
                 classifier_act='relu',
                 classifier_norm=None,
                 # Other parameters
                 lag_time=1):
        """
        Initialize the VAMPNet with encoder, classifier, and VAMP score.

        Parameters
        ----------
        encoder : nn.Module
            The encoder network for dimensionality reduction
        vamp_score : nn.Module
            Module for calculating VAMP scores

        embedding_module : nn.Module, optional
            Custom embedding module (takes precedence if provided)
        embedding_in_dim : int, optional
            Input dimension for built-in embedding MLP
        embedding_out_dim : int, optional
            Output dimension for built-in embedding MLP
        embedding_hidden_dim : int, default=64
            Hidden dimension for built-in embedding MLP
        embedding_num_layers : int, default=2
            Number of layers for built-in embedding MLP
        embedding_dropout : float, default=0.0
            Dropout rate for built-in embedding MLP
        embedding_act : str or callable, default='relu'
            Activation function for built-in embedding MLP
        embedding_norm : str or callable, default=None
            Normalization for built-in embedding MLP

        classifier_module : nn.Module, optional
            Custom classifier module (takes precedence if provided)
        n_classes : int, optional
            Number of output classes for built-in classifier
        classifier_hidden_dim : int, default=64
            Hidden dimension for built-in classifier
        classifier_num_layers : int, default=2
            Number of layers for built-in classifier
        classifier_dropout : float, default=0.0
            Dropout rate for built-in classifier
        classifier_act : str or callable, default='relu'
            Activation function for built-in classifier
        classifier_norm : str or callable, default=None
            Normalization for built-in classifier

        lag_time : int, default=1
            Lag time for time-lagged datasets
        """
        super(VAMPNet, self).__init__()

        self.encoder = encoder
        self.vamp_score = vamp_score
        self.lag_time = lag_time

        # Set up embedding module if needed
        if embedding_module is not None:
            # Use custom embedding module
            self.embedding_module = embedding_module
        elif embedding_in_dim is not None and embedding_out_dim is not None:
            # Create built-in MLP embedding
            self.embedding_module = MLP(
                in_channels=embedding_in_dim,
                hidden_channels=embedding_hidden_dim,
                out_channels=embedding_out_dim,
                num_layers=embedding_num_layers,
                dropout=embedding_dropout,
                act=embedding_act,
                norm=embedding_norm
            )
        else:
            # No embedding needed
            self.embedding_module = None

        # Handle classifier setup
        if classifier_module is not None:
            # User provided a custom classifier module
            self.classifier_module = classifier_module
        elif n_classes is not None:
            # Create built-in SoftmaxMLP classifier
            encoder_output_dim = getattr(encoder, 'output_dim', None)
            if encoder_output_dim is None:
                # Try to infer from hidden_dim if output_dim not found
                encoder_output_dim = getattr(encoder, 'hidden_dim', 64)

            # Create SoftmaxMLP classifier
            self.classifier_module = SoftmaxMLP(
                in_channels=encoder_output_dim,
                hidden_channels=classifier_hidden_dim,
                out_channels=n_classes,
                num_layers=classifier_num_layers,
                dropout=classifier_dropout,
                act=classifier_act,
                norm=classifier_norm
            )
        else:
            # No classifier specified
            self.classifier_module = None

    def forward(
        self,
        data,
        return_features: bool = False,
        apply_classifier: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Transform input data and optionally apply classifier.

        Args:
            data: Input data (can be graph data or tensor)
            return_features: Whether to return encoder features
            apply_classifier: Whether to apply classifier to get class probabilities

        Returns:
            If apply_classifier is True:
                If return_features is True: (class_probs, features)
                Else: class_probs
            Else:
                features
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
            # If encoder returns a tuple (like SchNet), extract the first element (graph embeddings)
            features = self.encoder(embedded_data.x, embedded_data.edge_index, embedded_data.edge_attr,
                                    embedded_data.batch)
            if isinstance(features, tuple):
                features = features[0]
        else:
            # For non-graph tensor data
            if self.embedding_module is not None:
                # Apply embedding then encoder
                return self.encoder(self.embedding_module(data))
            else:
                # Apply encoder directly
                return self.encoder(data)

        # Apply classifier if requested
        if apply_classifier and self.classifier_module is not None:
            probs = self.classifier_module(features)
            if return_features:
                return probs, features
            else:
                return probs, torch.empty(1)
        else:
            return torch.empty(1), features

    def transform(self, data, return_features=False):
        """
        Transform data into state probabilities.

        Parameters
        ----------
        data : torch.Tensor or torch_geometric.data.Data
            Input data
        return_features : bool, default=False
            Whether to also return encoder features

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If return_features is True: (probs, features)
            Else: probs
        """
        if self.classifier_module is None:
            raise ValueError("Classifier module is not defined. Cannot transform data.")

        return self.forward(data, return_features=return_features, apply_classifier=True)

    # TODO: Implement correctly (not working yet)
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

    # TODO: Implement correctly (not working yet)
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
