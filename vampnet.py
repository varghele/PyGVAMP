import torch
import torch.nn as nn
import torch_geometric as pyg


class VAMPNet(nn.Module):
    def __init__(self, encoder, lag_time=1):
        """
        Initialize the VAMPNet with a custom encoder.

        Args:
            encoder: The encoder network that transforms the input data
            lag_time: Lag time for the VAMP scoring
        """
        super(VAMPNet, self).__init__()

        self.encoder = encoder
        self.lag_time = lag_time

    def forward(self, x):
        """
        Transform input data using the encoder network.

        Args:
            x: Input data

        Returns:
            Transformed features
        """
        return self.encoder(x)

    def vamp_score(self, x_t0, x_t1, k=None):
        """
        Calculate the VAMP score between time-lagged data.

        Args:
            x_t0: Data at time t
            x_t1: Data at time t+lag_time
            k: Number of singular values to consider (if None, use all)

        Returns:
            VAMP score
        """
        # Apply feature transformation
        chi_t0 = self.encoder(x_t0)
        chi_t1 = self.encoder(x_t1)

        # Center the data
        chi_t0_mean = chi_t0.mean(dim=0, keepdim=True)
        chi_t1_mean = chi_t1.mean(dim=0, keepdim=True)
        chi_t0 = chi_t0 - chi_t0_mean
        chi_t1 = chi_t1 - chi_t1_mean

        # Calculate covariance matrices
        n_samples = x_t0.shape[0]
        cov_00 = (chi_t0.T @ chi_t0) / (n_samples - 1)
        cov_11 = (chi_t1.T @ chi_t1) / (n_samples - 1)
        cov_01 = (chi_t0.T @ chi_t1) / (n_samples - 1)

        # Regularize covariance matrices for numerical stability
        eps = 1e-6
        cov_00 += eps * torch.eye(cov_00.shape[0], device=cov_00.device)
        cov_11 += eps * torch.eye(cov_11.shape[0], device=cov_11.device)

        # Calculate VAMP matrix
        cov_00_inv_sqrt = self._matrix_inv_sqrt(cov_00)
        cov_11_inv_sqrt = self._matrix_inv_sqrt(cov_11)
        vamp_matrix = cov_00_inv_sqrt @ cov_01 @ cov_11_inv_sqrt

        # SVD
        U, S, Vh = torch.linalg.svd(vamp_matrix, full_matrices=False)

        # Take top k singular values if specified
        if k is not None:
            S = S[:k]

        # VAMP-2 score is the sum of squared singular values
        score = torch.sum(S ** 2)

        return score

    def _matrix_inv_sqrt(self, matrix):
        """Calculate the inverse square root of a matrix"""
        U, S, Vh = torch.linalg.svd(matrix)
        return U @ torch.diag(1.0 / torch.sqrt(S)) @ Vh

    def vamp_loss(self, x_t0, x_t1, k=None):
        """
        Calculate the VAMP loss (negative VAMP score).

        Args:
            x_t0: Data at time t
            x_t1: Data at time t+lag_time
            k: Number of singular values to consider

        Returns:
            VAMP loss
        """
        return -self.vamp_score(x_t0, x_t1, k)

    def create_time_lagged_dataset(self, data, lag_time=None):
        """
        Create a time-lagged dataset from sequential data.

        Args:
            data: Sequential data of shape (sequence_length, feature_dim)
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

    def fit(self, data_loader, optimizer, n_epochs=100, k=None):
        """
        Train the VAMPNet model.

        Args:
            data_loader: DataLoader providing batches of (x_t0, x_t1)
            optimizer: PyTorch optimizer
            n_epochs: Number of training epochs
            k: Number of singular values to consider for VAMP score

        Returns:
            List of loss values during training
        """
        losses = []

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in data_loader:
                x_t0, x_t1 = batch

                # Zero gradients
                optimizer.zero_grad()

                # Calculate VAMP loss
                loss = self.vamp_loss(x_t0, x_t1, k)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_epoch_loss = epoch_loss / n_batches
            losses.append(avg_epoch_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{n_epochs}, Loss: {avg_epoch_loss:.4f}")

        return losses
