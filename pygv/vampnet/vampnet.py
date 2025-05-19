import torch
import torch.nn as nn
from typing import Union, Tuple
from pygv.classifier.SoftmaxMLP import SoftmaxMLP
from torch_geometric.nn.models import MLP
import os
import datetime
from pygv.utils.plotting import plot_vamp_scores
from pygv.utils.nn_utils import monitor_gradients
from tqdm import tqdm


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
                 embedding_act='elu',
                 embedding_norm=None,
                 # Classifier options
                 classifier_module=None,
                 n_classes=None,
                 classifier_hidden_dim=64,
                 classifier_num_layers=2,
                 classifier_dropout=0.0,
                 classifier_act='elu',
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

        lag_time : float, default=1
            Lag time in ns, only a logging parameter
        """
        super(VAMPNet, self).__init__()

        self.encoder = encoder
        self.vamp_score = vamp_score
        self.lag_time = lag_time
        self.optimizer = None

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

        #self.add_module('embedding_module', embedding_module)
        self.add_module('encoder', encoder)
        self.add_module('classifier_module', classifier_module)

    def forward(
            self,
            data,
            return_features: bool = False,
            apply_classifier: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Transform input data through the model, with improved gradient flow.

        Args:
            data: Input PyG graph data object
            return_features: Whether to return encoder features
            apply_classifier: Whether to apply classifier to get class probabilities

        Returns:
            If apply_classifier is True:
                If return_features is True: (class_probs, features)
                Else: class_probs
            Else:
                features
        """
        # Extract graph components
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch if hasattr(data, 'batch') else None

        # Apply embedding module to node and edge features if provided
        if self.embedding_module is not None:
            # Add a skip connection for better gradient flow
            if hasattr(self.embedding_module, 'node_embedding') and hasattr(self.embedding_module, 'edge_embedding'):
                # Module has separate functions for node and edge embeddings
                embedded_x = self.embedding_module.node_embedding(x)
                x = embedded_x  # Direct assignment with no skip connection for node features

                if edge_attr is not None and self.embedding_module.edge_embedding is not None:
                    # Add skip connection for edge features if dimensions match
                    edge_embedded = self.embedding_module.edge_embedding(edge_attr)
                    if edge_embedded.size(-1) == edge_attr.size(-1):
                        edge_attr = edge_embedded + edge_attr  # Skip connection
                    else:
                        edge_attr = edge_embedded  # Direct assignment if dimensions don't match
            else:
                # Assume single embedding function for nodes
                x = self.embedding_module(x)

        # Track if we have a batch dimension
        has_batch = batch is not None

        # Ensure proper batch dimension if missing
        if not has_batch and len(x) > 1:
            # Create a batch vector with all zeros (single graph)
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)

        # Add a small jitter to prevent identical representations (helps with gradient flow)
        if self.training and x.requires_grad:
            # Add small noise (1e-6) during training
            x = x + torch.randn_like(x) * 1e-6

        try:
            # Pass through encoder
            features = self.encoder(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch
            )

            # Handle case where encoder returns a tuple
            if isinstance(features, tuple):
                features = features[0]

        except RuntimeError as e:
            # Provide helpful error messages for common issues
            if "Expected tensor for argument" in str(e):
                raise RuntimeError(f"Encoder error - check input dimensions: {str(e)}\n"
                                   f"x shape: {x.shape}, edge_index shape: {edge_index.shape}, "
                                   f"batch: {'is None' if batch is None else batch.shape}")
            elif "CUDA out of memory" in str(e):
                raise RuntimeError(f"CUDA out of memory in encoder. Try reducing batch size.")
            else:
                # Re-raise the original error
                raise

        # Apply classifier if requested
        if apply_classifier and self.classifier_module is not None:
            # Use try-except for better error messages
            try:
                probs = self.classifier_module(features)

                # Check for NaN values
                if torch.isnan(probs).any():
                    print("Warning: NaN values detected in classifier output")
                    # Replace NaN with small values to avoid breaking the loss
                    probs = torch.nan_to_num(probs, nan=1e-6)

                if return_features:
                    return probs, features
                else:
                    return probs, torch.empty(1)
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    raise RuntimeError(f"Size mismatch in classifier: expected input of shape "
                                       f"{getattr(self.classifier_module, 'in_channels', 'unknown')}, "
                                       f"but got {features.shape}")
                else:
                    raise
        else:
            return torch.empty(1), features

    def forward_old(
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
            features = self.encoder(embedded_data.x, embedded_data.edge_index, embedded_data.edge_attr, embedded_data.batch)
            if isinstance(features, tuple):
                features = features[0]
        else:
            # For non-graph tensor data #TODO: Note: This is probably redundant/not wanted
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

    def get_attention(self, data, device=None):
        """
        Extract attention maps from the encoder for the given data.

        This function runs the forward pass up to the encoder step and returns
        the attention matrices if available.

        Parameters
        ----------
        data : torch_geometric.data.Data or Batch
            PyTorch Geometric data object containing graph information
        device : str or torch.device, optional
            Device to run computation on. If None, uses the model's current device

        Returns
        -------
        tuple
            (features, attentions) where:
            - features: The encoded features/embedding
            - attentions: List of attention matrices or None if attention isn't used
        """
        # Set device
        if device is None:
            device = next(self.parameters()).device

        # Move data to device if needed
        if data.x.device != device:
            data = data.to(device)

        # Set model to eval mode for attention extraction
        was_training = self.training
        self.eval()

        with torch.no_grad():
            # Extract graph components
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
            batch = data.batch if hasattr(data, 'batch') else None

            # Apply embedding module if available
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

            # Pass through encoder to get features and attention
            encoder_output = self.encoder(x, edge_index, edge_attr, batch)

            # Handle different return types from the encoder
            if isinstance(encoder_output, tuple) and len(encoder_output) > 1:
                features, attention_info = encoder_output

                # Check if attention_info is a tuple containing attention
                if isinstance(attention_info, tuple) and len(attention_info) > 1:
                    _, attention_maps = attention_info
                    attentions = attention_maps
                else:
                    attentions = attention_info
            else:
                features = encoder_output
                attentions = None

        # Restore training mode
        if was_training:
            self.train()

        return features, attentions

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

    def save(self, filepath, save_optimizer=False, optimizer=None, metadata=None):
        """
        Save the VAMPNet model to disk, including all components and optional metadata.

        Parameters
        ----------
        filepath : str
            Path to save the model
        save_optimizer : bool, default=False
            Whether to save optimizer state
        optimizer : torch.optim.Optimizer, optional
            Optimizer to save if save_optimizer is True
        metadata : dict, optional
            Additional metadata to save with the model (e.g., training parameters, dataset info)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Prepare components to save
        save_dict = {
            'model_state_dict': self.state_dict(),
            'encoder_type': type(self.encoder).__name__,
            'encoder_params': getattr(self.encoder, 'params', {}),
            'vamp_score_config': {
                'epsilon': getattr(self.vamp_score, 'epsilon', 1e-6),
                'mode': getattr(self.vamp_score, 'mode', 'regularize')
            },
            'has_embedding': self.embedding_module is not None,
            'has_classifier': self.classifier_module is not None,
            'lag_time': self.lag_time,
            'timestamp': datetime.datetime.now().isoformat()
        }

        # Add optimizer state if requested
        if save_optimizer and optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
            save_dict['optimizer_type'] = type(optimizer).__name__

        # Add metadata
        if metadata is not None:
            save_dict['metadata'] = metadata

        # Save architecture-specific details
        if self.embedding_module is not None:
            save_dict['embedding_type'] = type(self.embedding_module).__name__

            # Save MLP configuration if using standard MLP
            if isinstance(self.embedding_module, MLP):
                save_dict['embedding_config'] = {
                    'in_channels': self.embedding_module.in_channels,
                    'hidden_channels': self.embedding_module.hidden_channels,
                    'out_channels': self.embedding_module.out_channels,
                    'num_layers': self.embedding_module.num_layers,
                }

        if isinstance(self.classifier_module, SoftmaxMLP):
            # For SoftmaxMLP
            # Check if mlp exists and get hidden_channels if it does
            mlp_hidden_channels = None
            if hasattr(self.classifier_module, 'mlp') and self.classifier_module.mlp is not None:
                mlp_hidden_channels = getattr(self.classifier_module.mlp, 'hidden_channels', None)

            # Get output dimension from final layer
            # The Sequential contains Linear + Softmax, so we need to access the Linear part
            if isinstance(self.classifier_module.final_layer, nn.Sequential):
                # Find the Linear layer in the Sequential
                for module in self.classifier_module.final_layer:
                    if isinstance(module, nn.Linear):
                        out_channels = module.out_features
                        break
                else:
                    # Fallback if no Linear layer found
                    out_channels = None
            elif isinstance(self.classifier_module.final_layer, nn.Linear):
                # Direct Linear layer
                out_channels = self.classifier_module.final_layer.out_features
            else:
                out_channels = None

            # Create config dictionary
            save_dict['classifier_config'] = {
                'in_channels': getattr(self.classifier_module, 'in_channels', None),
                'hidden_channels': mlp_hidden_channels,
                'out_channels': out_channels,
                'num_layers': getattr(self.classifier_module, 'num_layers', None),
            }

        # Save to file
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath, encoder_class, vamp_score_class,
             embedding_class=None, classifier_class=None, map_location=None):
        """
        Load a VAMPNet model from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model
        encoder_class : class
            Class of the encoder network
        vamp_score_class : class
            Class of the VAMP score module
        embedding_class : class, optional
            Class of the embedding module if used
        classifier_class : class, optional
            Class of the classifier module if used
        map_location : str or torch.device, optional
            Device to load the model to

        Returns
        -------
        tuple
            (loaded_model, metadata)
        """
        # Load saved dictionary
        try:
            checkpoint = torch.load(filepath, map_location=map_location)
        except Exception as e:
            raise IOError(f"Error loading model from {filepath}: {str(e)}")

        # Extract metadata
        metadata = checkpoint.get('metadata', {})

        try:
            # Create VAMP score module
            vamp_score_config = checkpoint.get('vamp_score_config', {})
            vamp_score = vamp_score_class(
                epsilon=vamp_score_config.get('epsilon', 1e-6),
                mode=vamp_score_config.get('mode', 'regularize')
            )

            # Create encoder
            encoder_params = checkpoint.get('encoder_params', {})
            encoder = encoder_class(**encoder_params)

            # Create embedding module if needed
            embedding_module = None
            if checkpoint.get('has_embedding', False):
                if embedding_class is None:
                    print("Warning: Model was saved with an embedding module, but no embedding_class provided.")
                else:
                    if 'embedding_config' in checkpoint and (embedding_class == MLP or
                                                             hasattr(embedding_class, 'in_channels')):
                        # Create embedding with saved config
                        config = checkpoint['embedding_config']
                        try:
                            embedding_module = embedding_class(
                                in_channels=config.get('in_channels'),
                                hidden_channels=config.get('hidden_channels'),
                                out_channels=config.get('out_channels'),
                                num_layers=config.get('num_layers', 2)
                            )
                        except TypeError as e:
                            print(f"Warning: Could not create embedding with saved config: {str(e)}")
                            print("Creating with default parameters instead.")
                            embedding_module = embedding_class()
                    else:
                        # Create custom embedding with default params
                        embedding_module = embedding_class()

            # Create classifier module if needed
            classifier_module = None
            if checkpoint.get('has_classifier', False):
                if classifier_class is None:
                    print("Warning: Model was saved with a classifier module, but no classifier_class provided.")
                else:
                    if 'classifier_config' in checkpoint and classifier_class == SoftmaxMLP:
                        # Create SoftmaxMLP classifier with saved config
                        config = checkpoint['classifier_config']
                        try:
                            classifier_module = classifier_class(
                                in_channels=config.get('in_channels'),
                                hidden_channels=config.get('hidden_channels'),
                                out_channels=config.get('out_channels'),
                                num_layers=config.get('num_layers', 2)
                            )
                        except TypeError as e:
                            print(f"Warning: Could not create classifier with saved config: {str(e)}")
                            print("Creating with default parameters instead.")
                            classifier_module = classifier_class()
                    else:
                        # Create custom classifier with default params
                        classifier_module = classifier_class()

            # Get lag time with fallback
            lag_time = checkpoint.get('lag_time', 1)

            # Create VAMPNet model
            model = cls(
                encoder=encoder,
                vamp_score=vamp_score,
                embedding_module=embedding_module,
                classifier_module=classifier_module,
                lag_time=lag_time
            )

            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])

            print(f"Model successfully loaded from {filepath}")
            return model, metadata

        except KeyError as e:
            raise KeyError(f"Missing key in checkpoint file: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error reconstructing model: {str(e)}")

    def save_complete_model(self, filepath):
        """
        Save the complete VAMPNet model to disk.

        This method saves the entire model structure (including all components)
        without requiring component classes during loading.

        Parameters
        ----------
        filepath : str
            Path to save the complete model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save the complete model with torch.save
        torch.save(self, filepath)
        print(f"Complete model saved to {filepath}")

    @staticmethod
    def load_complete_model(filepath, map_location=None):
        """
        Load a complete VAMPNet model from disk.

        This method loads the entire model structure as it was saved,
        without requiring component classes to be specified.

        Parameters
        ----------
        filepath : str
            Path to the saved complete model
        map_location : str or torch.device, optional
            Device to load the model to

        Returns
        -------
        VAMPNet
            Loaded model
        """
        try:
            # Load the complete model
            model = torch.load(filepath, map_location=map_location)
            print(f"Complete model loaded from {filepath}")
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model from {filepath}: {str(e)}")

    def get_config(self):
        """
        Get the configuration of the VAMPNet model for reproducibility.

        Returns
        -------
        dict
            Configuration dictionary
        """
        config = {
            'encoder_type': type(self.encoder).__name__,
            'vamp_score_type': type(self.vamp_score).__name__,
            'embedding_type': type(self.embedding_module).__name__ if self.embedding_module else None,
            'classifier_type': type(self.classifier_module).__name__ if self.classifier_module else None,
            'lag_time': self.lag_time
        }

        # Add encoder details
        if hasattr(self.encoder, 'get_config'):
            config['encoder_config'] = self.encoder.get_config()
        elif hasattr(self.encoder, 'params'):
            config['encoder_config'] = self.encoder.params

        # Add embedding details
        if isinstance(self.embedding_module, MLP):
            config['embedding_config'] = {
                'in_channels': self.embedding_module.in_channels,
                'hidden_channels': self.embedding_module.hidden_channels,
                'out_channels': self.embedding_module.out_channels,
                'num_layers': self.embedding_module.num_layers,
            }

        # Add classifier details
        if isinstance(self.classifier_module, SoftmaxMLP):
            config['classifier_config'] = {
                'in_channels': getattr(self.classifier_module, 'in_channels', None),
                'hidden_channels': getattr(self.classifier_module.mlp, 'hidden_channels', None)
                if getattr(self.classifier_module, 'mlp', None) else None,
                'out_channels': self.classifier_module.final_layer.out_features,
                'num_layers': getattr(self.classifier_module, 'num_layers', None),
            }

        return config

    def fit(
            self,
            train_loader,
            test_loader=None,
            optimizer=None,
            n_epochs=100,
            device=None,
            learning_rate=0.001,
            weight_decay=1e-5,
            save_dir="models",
            save_every=None,
            clip_grad_norm=None,
            verbose=True,
            show_batch_vamp=False,
            check_grad_stats=False,
            plot_scores=True,
            plot_path=None,
            smoothing=5,
            sample_validate_every=100,  # Validate on a sample batch every N batches
            early_stopping=None,  # Number of epochs with no improvement to trigger early stopping
            callbacks=None
    ):
        """
        Train the VAMPNet model using the provided data loader.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader providing batches of (x_t0, x_t1) time-lagged pairs for training
        test_loader : DataLoader, optional
            DataLoader providing batches of (x_t0, x_t1) time-lagged pairs for validation
        optimizer : torch.optim.Optimizer, optional
            PyTorch optimizer. If None, Adam optimizer will be created
        n_epochs : int, default=100
            Number of training epochs
        device : str or torch.device, optional
            Device to train on. If None, will use CUDA if available, else CPU
        learning_rate : float, default=0.001
            Learning rate for optimizer (if optimizer is None)
        weight_decay : float, default=1e-5
            Weight decay for optimizer (if optimizer is None)
        save_dir : str, default="models"
            Directory to save model checkpoints
        save_every : int, optional
            Save model every N epochs. If None, only saves final model
        clip_grad_norm : float, optional
            Maximum norm for gradient clipping. If None, no clipping is performed
        verbose : bool, default=True
            Whether to print training progress
        show_batch_vamp : bool, default=False
            Print VAMP score during each batch iteration
        check_grad_stats : bool, default=False
            Print gradient information
        plot_scores : bool, default=True
            Whether to plot training and validation scores after training
        plot_path : str, optional
            Path to save the score plots. If None, uses save_dir/vampnet_training_scores.png
        smoothing : int, default=5
            Window size for smoothing the score plots
        sample_validate_every : int, default=100
            Check validation performance on a single batch every N training batches
        early_stopping : int, optional
            Number of epochs without improvement after which to stop training
        callbacks : list, optional
            List of callback functions to call after each epoch

        Returns
        -------
        dict
            Dictionary containing training history
        """
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        # Move model to device
        self.to(device)

        # Create optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        self.optimizer = optimizer

        # Create save directory if it doesn't exist
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Helper function to move batch to device
        def to_device(batch, device_):
            x_t0, x_t1 = batch
            return (x_t0.to(device_), x_t1.to(device_))

        # Training history
        history = {
            'train_scores': [],  # Training scores for each epoch
            'batch_train_scores': [],  # Training scores for selected batches
            'batch_val_scores': [],  # Validation scores on samples
            'batch_indices': [],  # Corresponding batch indices
            'epoch_val_scores': [],  # Full validation scores per epoch
            'epochs': []  # Epoch numbers
        }

        # Initialize variables
        best_score = float('-inf')
        best_model_state = None
        no_improvement_count = 0
        global_batch = 1

        if verbose:
            print(f"Starting training for {n_epochs} epochs on {device}")
            if test_loader is not None:
                print(f"Using quick validation every {sample_validate_every} batches")
                print(f"Performing full validation after each epoch")
            if early_stopping:
                print(f"Early stopping after {early_stopping} epochs without improvement")

        # Training loop over epochs
        for epoch in range(n_epochs):
            # Set model to train mode
            self.train()

            epoch_score_sum = 0.0
            n_batches = 0

            # Use tqdm for progress bar if verbose
            iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=True) if verbose else train_loader

            # Process batches
            for batch_idx, batch in enumerate(iterator):
                # Move batch to device
                data_t0, data_t1 = to_device(batch, device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                chi_t0, _ = self.forward(data_t0, apply_classifier=True)
                chi_t1, _ = self.forward(data_t1, apply_classifier=True)

                # Calculate VAMP loss (negative VAMP score)
                loss = self.vamp_score.loss(chi_t0, chi_t1)

                # Get positive VAMP score for logging
                vamp_score_val = -loss.item()

                # Check for NaN loss
                if torch.isnan(loss).any():
                    if verbose:
                        print(f"Warning: NaN loss detected in epoch {epoch + 1}, batch {batch_idx}")
                    continue

                # Backward pass
                loss.backward()

                if check_grad_stats and batch_idx % 500 == 0:
                    from pygv.utils.analysis import monitor_gradients
                    grad_stats = monitor_gradients(self, epoch, batch_idx=batch_idx)
                    if grad_stats:
                        if grad_stats['max'] > 10.0:
                            print("⚠️ WARNING: Potential exploding gradients detected!")
                        if grad_stats['small_percent'] > 50.0:
                            print("⚠️ WARNING: Potential vanishing gradients detected!")

                # Gradient clipping if requested
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad_norm)

                # Optimizer step
                optimizer.step()

                # Update metrics
                epoch_score_sum += vamp_score_val
                n_batches += 1

                # Show batch VAMP score if requested
                if show_batch_vamp and batch_idx % 10 == 0:
                    current_avg = epoch_score_sum / n_batches
                    print(f"Batch {batch_idx}/{len(train_loader)}, VAMP: {vamp_score_val:.4f}, Avg: {current_avg:.4f}")

                # Perform quick validation check at regular intervals
                if test_loader is not None and global_batch % sample_validate_every == 0:
                    sample_val_score = self.quick_evaluate(test_loader, device)

                    if sample_val_score is not None:
                        # Store scores for this batch
                        history['batch_train_scores'].append(vamp_score_val)
                        history['batch_val_scores'].append(sample_val_score)
                        history['batch_indices'].append(global_batch)

                        # Update in progress bar
                        if verbose:
                            iterator.set_postfix({
                                'train_vamp': f"{vamp_score_val:.4f}",
                                'sample_val': f"{sample_val_score:.4f}"
                            })

                global_batch += 1

            # End of epoch

            # Calculate average train score for this epoch
            avg_train_score = epoch_score_sum / max(1, n_batches)
            history['train_scores'].append(avg_train_score)
            history['epochs'].append(epoch)

            # Perform full validation after each epoch if test loader exists
            current_val_score = None
            if test_loader is not None:
                current_val_score = self.evaluate(test_loader, device)
                history['epoch_val_scores'].append(current_val_score)

                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{n_epochs}, Train VAMP: {avg_train_score:.4f}, Val VAMP: {current_val_score:.4f}")
            else:
                # Use training score if no validation data
                current_val_score = avg_train_score
                if verbose:
                    print(f"Epoch {epoch + 1}/{n_epochs}, Train VAMP: {avg_train_score:.4f}")

            # Check if this is the best model
            if current_val_score > best_score:
                best_score = current_val_score
                no_improvement_count = 0

                # Save best model state
                best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}

                # Also save to disk if directory specified
                if save_dir:
                    self.save_complete_model(os.path.join(save_dir, "best_model.pt"))

                if verbose:
                    print(f"New best model with score: {best_score:.4f}")
            else:
                no_improvement_count += 1
                if verbose:
                    print(f"No improvement for {no_improvement_count} epochs. Best score: {best_score:.4f}")

            # Early stopping check
            if early_stopping and no_improvement_count >= early_stopping:
                print(f"Early stopping triggered after {no_improvement_count} epochs without improvement")
                break

            # Save checkpoint if requested
            if save_every and (epoch + 1) % save_every == 0 and save_dir:
                self.save_complete_model(os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt"))

            # Execute callbacks if provided
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch, current_val_score)

        # Save final model
        if save_dir:
            self.save_complete_model(os.path.join(save_dir, "final_model.pt"))

        # Load best model if we have one and early stopping was used
        if best_model_state is not None and no_improvement_count > 0:
            self.load_state_dict(best_model_state)
            print(f"Loaded best model with score: {best_score:.4f}")

        # Add best score to history
        history['best_score'] = best_score
        history['best_epoch'] = len(
            history['train_scores']) - no_improvement_count - 1 if no_improvement_count > 0 else len(
            history['train_scores']) - 1

        # Plot training performance if requested
        if plot_scores and plot_path:
            # Import here to avoid circular imports
            try:
                plot_vamp_scores(
                    history=history,
                    save_path=plot_path,
                    smoothing=smoothing,
                    title=f"VAMPNet Training Performance (lag={self.lag_time})"
                )
            except Exception as e:
                print(f"Warning: Could not create training plot: {e}")

        return history


    def quick_evaluate(self, data_loader, device=None, num_batches=10, verbose=True):
        """
        Quickly evaluate model performance by sampling batches from a data loader.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader providing batches of (x_t0, x_t1) time-lagged pairs
        device : torch.device, optional
            Device to use for evaluation
        num_batches : int, default=10
            Number of batches to evaluate and average
        verbose : bool, default=True
            Whether to print the validation score to console

        Returns
        -------
        float
            Average VAMP score across sampled batches, or None if evaluation fails
        """
        try:
            if data_loader is None or len(data_loader) == 0:
                return None

            # Set device if not provided
            if device is None:
                device = next(self.parameters()).device

            # Helper function to move batch to device
            def to_device(batch, device_):
                x_t0, x_t1 = batch
                return (x_t0.to(device_), x_t1.to(device_))

            # Limit number of batches to data loader size
            num_batches = min(num_batches, len(data_loader))

            # Evaluate in eval mode with no grad
            self.eval()
            batch_scores = []
            batch_sizes = []

            # Get iterator for the data loader
            iterator = iter(data_loader)

            with torch.no_grad():
                # Process the specified number of batches
                for i in range(num_batches):
                    try:
                        # Get the next batch
                        test_batch = next(iterator)
                    except StopIteration:
                        # Restart iterator if we run out of batches
                        iterator = iter(data_loader)
                        test_batch = next(iterator)

                    # Move batch to device
                    test_t0, test_t1 = to_device(test_batch, device)
                    batch_size = test_t0.size(0) if hasattr(test_t0, 'size') else 0

                    # Forward pass
                    test_chi_t0, _ = self.forward(test_t0, apply_classifier=True)
                    test_chi_t1, _ = self.forward(test_t1, apply_classifier=True)

                    # Get VAMP score
                    test_loss = self.vamp_score.loss(test_chi_t0, test_chi_t1)
                    test_score = -test_loss.item()

                    batch_scores.append(test_score)
                    batch_sizes.append(batch_size)

            # Back to training mode
            self.train()

            # Compute average score
            if not batch_scores:
                if verbose:
                    print("\nNo batches were successfully evaluated")
                return None

            avg_score = sum(batch_scores) / len(batch_scores)

            # Print validation score to console if requested
            if verbose:
                batch_score_str = ", ".join([f"{score:.4f}" for score in batch_scores])
                print(f"\nQuick validation ({len(batch_scores)} batches): Average VAMP score = {avg_score:.4f}")
                print(f"Individual batch scores: [{batch_score_str}]")

            return avg_score

        except Exception as e:
            if verbose:
                print(f"\nError in quick validation: {str(e)}")
            return None

    def evaluate(self, data_loader, device=None):
        """
        Fully evaluate the model on the given data loader.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader providing batches of (x_t0, x_t1) time-lagged pairs
        device : torch.device, optional
            Device to use for evaluation

        Returns
        -------
        float
            Average VAMP score
        """
        if data_loader is None or len(data_loader) == 0:
            return None

        # Set device if not provided
        if device is None:
            device = next(self.parameters()).device

        # Helper function to move batch to device
        def to_device(batch, device_):
            x_t0, x_t1 = batch
            return (x_t0.to(device_), x_t1.to(device_))

        self.eval()
        test_score_sum = 0.0
        n_test_batches = 0

        with torch.no_grad():
            for test_batch in data_loader:
                # Move batch to device
                test_t0, test_t1 = to_device(test_batch, device)

                # Forward pass
                test_chi_t0, _ = self.forward(test_t0, apply_classifier=True)
                test_chi_t1, _ = self.forward(test_t1, apply_classifier=True)

                # Calculate VAMP score
                test_loss = self.vamp_score.loss(test_chi_t0, test_chi_t1)
                test_score = -test_loss.item()

                test_score_sum += test_score
                n_test_batches += 1

        # Return to training mode
        self.train()

        # Return average test score
        return test_score_sum / max(1, n_test_batches)

