import torch
import torch.nn as nn
from typing import Optional


class VAMPScore(nn.Module):
    def __init__(self, epsilon=1e-6):
        """
        Initialize the VAMP Score calculator.

        Args:
            epsilon: Regularization parameter for covariance matrices
        """
        super(VAMPScore, self).__init__()
        self.epsilon = epsilon

    def _matrix_inv_sqrt(self, matrix):
        """Calculate the inverse square root of a matrix"""
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        # Handle very small singular values
        mask = S > 1e-10
        S_inv_sqrt = torch.zeros_like(S)
        S_inv_sqrt[mask] = 1.0 / torch.sqrt(S[mask])
        return U @ torch.diag(S_inv_sqrt) @ Vh

    def forward(self, chi_t0, chi_t1, k=None, center=True):
        """
        Calculate the VAMP score between two sets of embeddings.

        Args:
            chi_t0: Embeddings at time t
            chi_t1: Embeddings at time t+lag_time
            k: Number of singular values to consider (if None, use all)
            center: Whether to center the data

        Returns:
            VAMP score (VAMP-2 score: sum of squared singular values)
        """
        if center:
            # Center the data
            chi_t0_mean = chi_t0.mean(dim=0, keepdim=True)
            chi_t1_mean = chi_t1.mean(dim=0, keepdim=True)
            chi_t0 = chi_t0 - chi_t0_mean
            chi_t1 = chi_t1 - chi_t1_mean

        # Calculate covariance matrices
        n_samples = chi_t0.shape[0]
        cov_00 = (chi_t0.T @ chi_t0) / (n_samples - 1)
        cov_11 = (chi_t1.T @ chi_t1) / (n_samples - 1)
        cov_01 = (chi_t0.T @ chi_t1) / (n_samples - 1)

        # Regularize covariance matrices for numerical stability
        cov_00 += self.epsilon * torch.eye(cov_00.shape[0], device=cov_00.device)
        cov_11 += self.epsilon * torch.eye(cov_11.shape[0], device=cov_11.device)

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
        vamp2_score = torch.sum(S ** 2)

        return vamp2_score

    def loss(self, chi_t0, chi_t1, k=None, center=True):
        """
        Calculate the VAMP loss (negative VAMP score).

        Args:
            chi_t0: Embeddings at time t
            chi_t1: Embeddings at time t+lag_time
            k: Number of singular values to consider
            center: Whether to center the data

        Returns:
            VAMP loss
        """
        return -self.forward(chi_t0, chi_t1, k, center)
