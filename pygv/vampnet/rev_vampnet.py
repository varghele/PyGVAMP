import torch
import torch.nn as nn
from typing import Union, Tuple
from pygv.classifier.SoftmaxMLP import SoftmaxMLP
from pygv.scores.reversible_score import ReversibleVAMPScore
from torch_geometric.nn.models import MLP
import os
import datetime
from pygv.utils.plotting import plot_vamp_scores
from pygv.utils.nn_utils import monitor_gradients
from tqdm import tqdm


class RevVAMPNet(nn.Module):
    """
    Reversible GraphVAMPNet model.

    Uses a likelihood-based loss with a learned transition matrix that
    satisfies detailed balance by construction. The network architecture
    (embedding → encoder → classifier → softmax) is identical to VAMPNet,
    but training uses ReversibleVAMPScore instead of VAMPScore.

    Parameters
    ----------
    encoder : nn.Module
        The encoder network for dimensionality reduction
    rev_score : ReversibleVAMPScore
        Reversible score module with learnable transition matrix
    embedding_module : nn.Module, optional
        Custom embedding module
    classifier_module : nn.Module, optional
        Custom classifier module
    lag_time : float, default=1
        Lag time in ns (logging parameter)
    training_jitter : float, default=1e-6
        Small noise added during training for gradient flow
    """

    def __init__(self,
                 encoder,
                 rev_score,
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
                 lag_time=1,
                 training_jitter=1e-6):
        super(RevVAMPNet, self).__init__()

        self.encoder = encoder
        self.rev_score = rev_score
        self.lag_time = lag_time
        self.training_jitter = training_jitter
        self.optimizer = None

        # Set up embedding module if needed
        if embedding_module is not None:
            self.embedding_module = embedding_module
        elif embedding_in_dim is not None and embedding_out_dim is not None:
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
            self.embedding_module = None

        # Handle classifier setup
        if classifier_module is not None:
            self.classifier_module = classifier_module
        elif n_classes is not None:
            encoder_output_dim = getattr(encoder, 'output_dim', None)
            if encoder_output_dim is None:
                encoder_output_dim = getattr(encoder, 'hidden_dim', 64)

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
            self.classifier_module = None

        self.add_module('encoder', encoder)
        self.add_module('classifier_module', classifier_module)

    def forward(
            self,
            data,
            return_features: bool = False,
            apply_classifier: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Transform input data through the model.

        Args:
            data: Input PyG graph data object
            return_features: Whether to return encoder features
            apply_classifier: Whether to apply classifier to get class probabilities

        Returns:
            If apply_classifier is True:
                If return_features is True: (class_probs, features)
                Else: (class_probs, empty_tensor)
            Else:
                (empty_tensor, features)
        """
        # Extract graph components
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch if hasattr(data, 'batch') else None

        # Apply embedding module to node and edge features if provided
        if self.embedding_module is not None:
            if hasattr(self.embedding_module, 'node_embedding') and hasattr(self.embedding_module, 'edge_embedding'):
                embedded_x = self.embedding_module.node_embedding(x)
                x = embedded_x

                if edge_attr is not None and self.embedding_module.edge_embedding is not None:
                    edge_embedded = self.embedding_module.edge_embedding(edge_attr)
                    if edge_embedded.size(-1) == edge_attr.size(-1):
                        edge_attr = edge_embedded + edge_attr
                    else:
                        edge_attr = edge_embedded
            else:
                x = self.embedding_module(x)

        # Track if we have a batch dimension
        has_batch = batch is not None

        # Ensure proper batch dimension if missing
        if not has_batch and len(x) > 1:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)

        # Add a small jitter to prevent identical representations
        if self.training and x.requires_grad:
            x = x + torch.randn_like(x) * self.training_jitter

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
            if "Expected tensor for argument" in str(e):
                raise RuntimeError(f"Encoder error - check input dimensions: {str(e)}\n"
                                   f"x shape: {x.shape}, edge_index shape: {edge_index.shape}, "
                                   f"batch: {'is None' if batch is None else batch.shape}")
            elif "CUDA out of memory" in str(e):
                raise RuntimeError(f"CUDA out of memory in encoder. Try reducing batch size.")
            else:
                raise

        # Apply classifier if requested
        if apply_classifier and self.classifier_module is not None:
            try:
                probs = self.classifier_module(features)

                # Check for NaN values
                if torch.isnan(probs).any():
                    print("Warning: NaN values detected in classifier output")
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

    def get_attention(self, data, device=None):
        """
        Extract attention maps from the encoder for the given data.

        Parameters
        ----------
        data : torch_geometric.data.Data or Batch
            PyTorch Geometric data object
        device : str or torch.device, optional
            Device to run computation on

        Returns
        -------
        tuple
            (features, attentions)
        """
        if device is None:
            device = next(self.parameters()).device

        if data.x.device != device:
            data = data.to(device)

        was_training = self.training
        self.eval()

        with torch.no_grad():
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
            batch = data.batch if hasattr(data, 'batch') else None

            if self.embedding_module is not None:
                if hasattr(self.embedding_module, 'node_embedding') and hasattr(self.embedding_module,
                                                                                'edge_embedding'):
                    x = self.embedding_module.node_embedding(x)
                    if edge_attr is not None and self.embedding_module.edge_embedding is not None:
                        edge_attr = self.embedding_module.edge_embedding(edge_attr)
                else:
                    x = self.embedding_module(x)

            encoder_output = self.encoder(x, edge_index, edge_attr, batch)

            if isinstance(encoder_output, tuple) and len(encoder_output) > 1:
                features, attention_info = encoder_output

                if isinstance(attention_info, tuple) and len(attention_info) > 1:
                    _, attention_maps = attention_info
                    attentions = attention_maps
                else:
                    attentions = attention_info
            else:
                features = encoder_output
                attentions = None

        if was_training:
            self.train()

        return features, attentions

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
            return self.embedding_module(data)

    def get_transition_matrix(self) -> torch.Tensor:
        """
        Get the learned transition matrix from the reversible score module.

        Returns
        -------
        torch.Tensor
            Row-stochastic transition matrix satisfying detailed balance
        """
        return self.rev_score.get_transition_matrix()

    def get_stationary_distribution(self) -> torch.Tensor:
        """
        Get the learned stationary distribution from the reversible score module.

        Returns
        -------
        torch.Tensor
            Stationary distribution (positive, sums to 1)
        """
        return self.rev_score.get_stationary_distribution()

    def save(self, filepath, save_optimizer=False, optimizer=None, metadata=None):
        """
        Save the RevVAMPNet model to disk.

        Parameters
        ----------
        filepath : str
            Path to save the model
        save_optimizer : bool, default=False
            Whether to save optimizer state
        optimizer : torch.optim.Optimizer, optional
            Optimizer to save
        metadata : dict, optional
            Additional metadata
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        save_dict = {
            'model_state_dict': self.state_dict(),
            'encoder_type': type(self.encoder).__name__,
            'encoder_params': getattr(self.encoder, 'params', {}),
            'rev_score_config': {
                'n_states': self.rev_score.n_states,
                'epsilon': self.rev_score.epsilon,
            },
            'has_embedding': self.embedding_module is not None,
            'has_classifier': self.classifier_module is not None,
            'lag_time': self.lag_time,
            'reversible': True,
            'timestamp': datetime.datetime.now().isoformat()
        }

        if save_optimizer and optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
            save_dict['optimizer_type'] = type(optimizer).__name__

        if metadata is not None:
            save_dict['metadata'] = metadata

        if self.embedding_module is not None:
            save_dict['embedding_type'] = type(self.embedding_module).__name__
            if isinstance(self.embedding_module, MLP):
                save_dict['embedding_config'] = {
                    'in_channels': self.embedding_module.in_channels,
                    'hidden_channels': self.embedding_module.hidden_channels,
                    'out_channels': self.embedding_module.out_channels,
                    'num_layers': self.embedding_module.num_layers,
                }

        if isinstance(self.classifier_module, SoftmaxMLP):
            mlp_hidden_channels = None
            if hasattr(self.classifier_module, 'mlp') and self.classifier_module.mlp is not None:
                mlp_hidden_channels = getattr(self.classifier_module.mlp, 'hidden_channels', None)

            if isinstance(self.classifier_module.final_layer, nn.Sequential):
                for module in self.classifier_module.final_layer:
                    if isinstance(module, nn.Linear):
                        out_channels = module.out_features
                        break
                else:
                    out_channels = None
            elif isinstance(self.classifier_module.final_layer, nn.Linear):
                out_channels = self.classifier_module.final_layer.out_features
            else:
                out_channels = None

            save_dict['classifier_config'] = {
                'in_channels': getattr(self.classifier_module, 'in_channels', None),
                'hidden_channels': mlp_hidden_channels,
                'out_channels': out_channels,
                'num_layers': getattr(self.classifier_module, 'num_layers', None),
            }

        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")

    def save_complete_model(self, filepath):
        """
        Save the complete RevVAMPNet model to disk.

        Parameters
        ----------
        filepath : str
            Path to save the complete model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self, filepath)
        print(f"Complete model saved to {filepath}")

    @staticmethod
    def load_complete_model(filepath, map_location=None):
        """
        Load a complete RevVAMPNet model from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved complete model
        map_location : str or torch.device, optional
            Device to load the model to

        Returns
        -------
        RevVAMPNet
            Loaded model
        """
        try:
            model = torch.load(filepath, map_location=map_location, weights_only=False)
            print(f"Complete model loaded from {filepath}")
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model from {filepath}: {str(e)}")

    def get_config(self):
        """
        Get the configuration of the RevVAMPNet model.

        Returns
        -------
        dict
            Configuration dictionary
        """
        config = {
            'encoder_type': type(self.encoder).__name__,
            'rev_score_type': type(self.rev_score).__name__,
            'embedding_type': type(self.embedding_module).__name__ if self.embedding_module else None,
            'classifier_type': type(self.classifier_module).__name__ if self.classifier_module else None,
            'lag_time': self.lag_time,
            'reversible': True,
        }

        if hasattr(self.encoder, 'get_config'):
            config['encoder_config'] = self.encoder.get_config()
        elif hasattr(self.encoder, 'params'):
            config['encoder_config'] = self.encoder.params

        if isinstance(self.embedding_module, MLP):
            config['embedding_config'] = {
                'in_channels': self.embedding_module.in_channels,
                'hidden_channels': self.embedding_module.hidden_channels,
                'out_channels': self.embedding_module.out_channels,
                'num_layers': self.embedding_module.num_layers,
            }

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
            scheduler=None,
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
            sample_validate_every=100,
            early_stopping=None,
            callbacks=None
    ):
        """
        Train the RevVAMPNet model using negative log-likelihood loss.

        Parameters are identical to VAMPNet.fit(). The key differences are:
        - Loss is NLL (lower is better), not negative VAMP score
        - Best model tracks lowest NLL instead of highest VAMP score
        - Score logging records NLL values

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

        self.to(device)

        # Create optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        self.optimizer = optimizer

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        def to_device(batch, device_):
            x_t0, x_t1 = batch
            return (x_t0.to(device_), x_t1.to(device_))

        # Training history
        history = {
            'train_scores': [],
            'batch_train_scores': [],
            'batch_val_scores': [],
            'batch_indices': [],
            'epoch_val_scores': [],
            'epochs': []
        }

        # For NLL, best score is lowest (start with +inf)
        best_score = float('inf')
        best_model_state = None
        no_improvement_count = 0
        global_batch = 1

        if verbose:
            print(f"Starting RevGraphVAMPNet training for {n_epochs} epochs on {device}")
            if test_loader is not None:
                print(f"Using quick validation every {sample_validate_every} batches")
                print(f"Performing full validation after each epoch")
            if early_stopping:
                print(f"Early stopping after {early_stopping} epochs without improvement")

        for epoch in range(n_epochs):
            self.train()

            epoch_loss_sum = 0.0
            n_batches = 0

            iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=True) if verbose else train_loader

            for batch_idx, batch in enumerate(iterator):
                data_t0, data_t1 = to_device(batch, device)

                optimizer.zero_grad()

                chi_t0, _ = self.forward(data_t0, apply_classifier=True)
                chi_t1, _ = self.forward(data_t1, apply_classifier=True)

                # NLL loss (lower is better)
                loss = self.rev_score.loss(chi_t0, chi_t1)
                nll_val = loss.item()

                if torch.isnan(loss).any():
                    if verbose:
                        print(f"Warning: NaN loss detected in epoch {epoch + 1}, batch {batch_idx}")
                    continue

                loss.backward()

                if check_grad_stats and batch_idx % 500 == 0:
                    grad_stats = monitor_gradients(self, epoch, batch_idx=batch_idx)
                    if grad_stats:
                        if grad_stats['max'] > 10.0:
                            print("WARNING: Potential exploding gradients detected!")
                        if grad_stats['small_percent'] > 50.0:
                            print("WARNING: Potential vanishing gradients detected!")

                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad_norm)

                optimizer.step()

                epoch_loss_sum += nll_val
                n_batches += 1

                if show_batch_vamp and batch_idx % 10 == 0:
                    current_avg = epoch_loss_sum / n_batches
                    print(f"Batch {batch_idx}/{len(train_loader)}, NLL: {nll_val:.4f}, Avg: {current_avg:.4f}")

                # Quick validation
                if test_loader is not None and global_batch % sample_validate_every == 0:
                    sample_val_nll = self.quick_evaluate(test_loader, device)

                    if sample_val_nll is not None:
                        history['batch_train_scores'].append(nll_val)
                        history['batch_val_scores'].append(sample_val_nll)
                        history['batch_indices'].append(global_batch)

                        if verbose:
                            iterator.set_postfix({
                                'train_nll': f"{nll_val:.4f}",
                                'sample_val_nll': f"{sample_val_nll:.4f}"
                            })

                global_batch += 1

            # End of epoch
            avg_train_nll = epoch_loss_sum / max(1, n_batches)
            history['train_scores'].append(avg_train_nll)
            history['epochs'].append(epoch)

            # Full validation
            current_val_nll = None
            if test_loader is not None:
                current_val_nll = self.evaluate(test_loader, device)
                history['epoch_val_scores'].append(current_val_nll)

                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{n_epochs}, Train NLL: {avg_train_nll:.4f}, Val NLL: {current_val_nll:.4f}")
            else:
                current_val_nll = avg_train_nll
                if verbose:
                    print(f"Epoch {epoch + 1}/{n_epochs}, Train NLL: {avg_train_nll:.4f}")

            # Best model tracking: lowest NLL is best
            if current_val_nll < best_score:
                best_score = current_val_nll
                no_improvement_count = 0

                best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}

                if save_dir:
                    self.save_complete_model(os.path.join(save_dir, "best_model.pt"))

                if verbose:
                    print(f"New best model with NLL: {best_score:.4f}")
            else:
                no_improvement_count += 1
                if verbose:
                    print(f"No improvement for {no_improvement_count} epochs. Best NLL: {best_score:.4f}")

            # Early stopping
            if early_stopping and no_improvement_count >= early_stopping:
                print(f"Early stopping triggered after {no_improvement_count} epochs without improvement")
                break

            if save_every and (epoch + 1) % save_every == 0 and save_dir:
                self.save_complete_model(os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt"))

            # Step LR scheduler (once per epoch) and log the new rate
            if scheduler is not None:
                scheduler.step()
                if verbose:
                    next_lr = optimizer.param_groups[0]['lr']
                    print(f"  LR now: {next_lr:.6g}")

            if callbacks:
                for callback in callbacks:
                    callback(self, epoch, current_val_nll)

        # Save final model
        if save_dir:
            self.save_complete_model(os.path.join(save_dir, "final_model.pt"))

        # Load best model
        if best_model_state is not None and no_improvement_count > 0:
            self.load_state_dict(best_model_state)
            print(f"Loaded best model with NLL: {best_score:.4f}")

        history['best_score'] = best_score
        history['best_epoch'] = len(
            history['train_scores']) - no_improvement_count - 1 if no_improvement_count > 0 else len(
            history['train_scores']) - 1

        # Plot training performance
        if plot_scores and plot_path:
            try:
                plot_vamp_scores(
                    history=history,
                    save_path=plot_path,
                    smoothing=smoothing,
                    title=f"RevGraphVAMPNet Training (lag={self.lag_time})"
                )
            except Exception as e:
                print(f"Warning: Could not create training plot: {e}")

        return history

    def quick_evaluate(self, data_loader, device=None, num_batches=10, verbose=True):
        """
        Quickly evaluate model performance by sampling batches.

        Returns
        -------
        float
            Average NLL across sampled batches, or None if evaluation fails
        """
        try:
            if data_loader is None or len(data_loader) == 0:
                return None

            if device is None:
                device = next(self.parameters()).device

            def to_device(batch, device_):
                x_t0, x_t1 = batch
                return (x_t0.to(device_), x_t1.to(device_))

            num_batches = min(num_batches, len(data_loader))

            # The reversible NLL is linear in samples, so per-batch averaging
            # is mathematically equivalent when batches are equal-sized. We
            # still concat-and-score once for consistency with the VAMP-2 path.
            self.eval()
            chi_t0_chunks = []
            chi_t1_chunks = []

            iterator = iter(data_loader)

            with torch.no_grad():
                for i in range(num_batches):
                    try:
                        test_batch = next(iterator)
                    except StopIteration:
                        iterator = iter(data_loader)
                        test_batch = next(iterator)

                    test_t0, test_t1 = to_device(test_batch, device)

                    test_chi_t0, _ = self.forward(test_t0, apply_classifier=True)
                    test_chi_t1, _ = self.forward(test_t1, apply_classifier=True)

                    chi_t0_chunks.append(test_chi_t0)
                    chi_t1_chunks.append(test_chi_t1)

            self.train()

            if not chi_t0_chunks:
                if verbose:
                    print("\nNo batches were successfully evaluated")
                return None

            chi_t0_full = torch.cat(chi_t0_chunks, dim=0)
            chi_t1_full = torch.cat(chi_t1_chunks, dim=0)

            with torch.no_grad():
                nll = self.rev_score.loss(chi_t0_full, chi_t1_full).item()

            if verbose:
                print(f"\nQuick validation ({len(chi_t0_chunks)} batches, "
                      f"{chi_t0_full.shape[0]} samples): NLL = {nll:.4f}")

            return nll

        except Exception as e:
            if verbose:
                print(f"\nError in quick validation: {str(e)}")
            return None

    def evaluate(self, data_loader, device=None):
        """
        Fully evaluate the model on the given data loader.

        Concatenates chi outputs across batches and scores once on the full
        tensor. The reversible NLL is linear in samples, so with equal-sized
        batches this matches per-batch averaging — the style mirrors the
        VAMP-2 evaluator, where concat-and-score-once is required for
        correctness.

        Returns
        -------
        float
            NLL computed on the full validation set.
        """
        if data_loader is None or len(data_loader) == 0:
            return None

        if device is None:
            device = next(self.parameters()).device

        def to_device(batch, device_):
            x_t0, x_t1 = batch
            return (x_t0.to(device_), x_t1.to(device_))

        self.eval()
        chi_t0_chunks = []
        chi_t1_chunks = []

        with torch.no_grad():
            for test_batch in data_loader:
                test_t0, test_t1 = to_device(test_batch, device)

                test_chi_t0, _ = self.forward(test_t0, apply_classifier=True)
                test_chi_t1, _ = self.forward(test_t1, apply_classifier=True)

                chi_t0_chunks.append(test_chi_t0)
                chi_t1_chunks.append(test_chi_t1)

        self.train()

        if not chi_t0_chunks:
            return None

        chi_t0_full = torch.cat(chi_t0_chunks, dim=0)
        chi_t1_full = torch.cat(chi_t1_chunks, dim=0)

        with torch.no_grad():
            nll = self.rev_score.loss(chi_t0_full, chi_t1_full).item()

        return nll
