import torch
import torch.nn as nn
from typing import Literal, Optional, Union, List


class VAMPScore(nn.Module):
    def __init__(
            self,
            method: Literal['VAMP1', 'VAMP2', 'VAMPE', 'VAMPCE'] = 'VAMP2',
            epsilon: float = 1e-6,
            mode: Literal['trunc', 'regularize'] = 'trunc'
    ):
        """
        VAMP Score module for evaluating time-lagged embeddings.

        Parameters
        ----------
        method : str, default='VAMP2'
            VAMP scoring method:
            - 'VAMP1': Nuclear norm of Koopman matrix
            - 'VAMP2': Squared Frobenius norm of Koopman matrix
            - 'VAMPE': Eigendecomposition-based score
            - 'VAMPCE': Custom scoring with cross-entropy
        epsilon : float, default=1e-6
            Cutoff parameter for small eigenvalues or regularization parameter
        mode : str, default='trunc'
            Mode for handling small eigenvalues:
            - 'trunc': Truncate eigenvalues smaller than epsilon
            - 'regularize': Add epsilon to diagonal for regularization
        """
        super(VAMPScore, self).__init__()

        self.method = method
        self.epsilon = epsilon
        self.mode = mode

        # Validate parameters
        valid_methods = ['VAMP1', 'VAMP2', 'VAMPE', 'VAMPCE']
        valid_modes = ['trunc', 'regularize']

        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{self.method}', supported are {valid_methods}")

        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}', supported are {valid_modes}")

    def forward(self, data: torch.Tensor, data_lagged: torch.Tensor) -> torch.Tensor:
        """
        Compute the VAMP score based on data and corresponding time-shifted data.

        Parameters
        ----------
        data : torch.Tensor
            (N, d)-dimensional torch tensor containing instantaneous data
        data_lagged : torch.Tensor
            (N, d)-dimensional torch tensor containing time-lagged data

        Returns
        -------
        torch.Tensor
            Computed VAMP score. Includes +1 contribution from constant singular function.
        """
        # Validate inputs
        if not torch.is_tensor(data) or not torch.is_tensor(data_lagged):
            raise TypeError("Data inputs must be torch.Tensor objects")

        if data.shape != data_lagged.shape:
            raise ValueError(f"Data shapes must match but were {data.shape} and {data_lagged.shape}")

        # Compute score based on method
        if self.method in ['VAMP1', 'VAMP2']:
            koopman = self._koopman_matrix(data, data_lagged)

            if self.method == 'VAMP1':
                score = torch.norm(koopman, p='nuc')
            else:  # VAMP2
                score = torch.pow(torch.norm(koopman, p='fro'), 2)

        elif self.method == 'VAMPE':
            # Compute Covariance
            c00, c0t, ctt = self._covariances(data, data_lagged, remove_mean=True)

            # Compute inverse square roots
            c00_sqrt_inv = self._sym_inverse(c00, return_sqrt=True)
            ctt_sqrt_inv = self._sym_inverse(ctt, return_sqrt=True)

            # Compute Koopman operator
            koopman = self._multi_dot([c00_sqrt_inv, c0t, ctt_sqrt_inv]).t()

            # SVD decomposition
            u, s, v = torch.svd(koopman)
            mask = s > self.epsilon

            # Apply mask and compute transformed matrices
            u = torch.mm(c00_sqrt_inv, u[:, mask])
            v = torch.mm(ctt_sqrt_inv, v[:, mask])
            s = s[mask]

            # Compute score
            u_t = u.t()
            v_t = v.t()
            s = torch.diag(s)
            score = torch.trace(
                2. * self._multi_dot([s, u_t, c0t, v]) -
                self._multi_dot([s, u_t, c00, u, s, v_t, ctt, v])
            )

        elif self.method == 'VAMPCE':
            score = torch.trace(data[0])
            score = -1.0 * score
            return score

        # Add the +1 contribution from constant singular function
        final_score = 1 + score

        return final_score

    def loss(self, data: torch.Tensor, data_lagged: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """
        Calculate the VAMP loss (negative VAMP score).

        Parameters
        ----------
        data : torch.Tensor
            (N, d)-dimensional torch tensor containing instantaneous data
        data_lagged : torch.Tensor
            (N, d)-dimensional torch tensor containing time-lagged data
        k : Optional[int], default=None
            Number of singular values to consider (not used in this implementation)

        Returns
        -------
        torch.Tensor
            VAMP loss
        """
        # The k parameter is not used in this implementation but kept for API compatibility
        return -self.forward(data, data_lagged)

    def _koopman_matrix(self, data: torch.Tensor, data_lagged: torch.Tensor) -> torch.Tensor:
        """
        Compute the Koopman matrix from data and time-lagged data.

        Parameters
        ----------
        data : torch.Tensor
            Instantaneous data
        data_lagged : torch.Tensor
            Time-lagged data

        Returns
        -------
        torch.Tensor
            Koopman matrix
        """
        # Compute covariances
        c00, c0t, ctt = self._covariances(data, data_lagged, remove_mean=True)

        # Compute inverse square roots
        c00_sqrt_inv = self._sym_inverse(c00, return_sqrt=True)
        ctt_sqrt_inv = self._sym_inverse(ctt, return_sqrt=True)

        # Compute Koopman matrix: C00^(-1/2) @ C0t @ Ctt^(-1/2)
        koopman = self._multi_dot([c00_sqrt_inv, c0t, ctt_sqrt_inv])

        return koopman

    def _covariances(self, data: torch.Tensor, data_lagged: torch.Tensor, remove_mean: bool = True) -> tuple:
        """
        Compute covariance matrices from data and time-lagged data.

        Parameters
        ----------
        data : torch.Tensor
            Instantaneous data
        data_lagged : torch.Tensor
            Time-lagged data
        remove_mean : bool, default=True
            Whether to remove the mean from the data

        Returns
        -------
        tuple(torch.Tensor, torch.Tensor, torch.Tensor)
            Tuple of covariance matrices (C00, C0t, Ctt)
        """
        n_samples = data.shape[0]

        # Center the data if requested
        if remove_mean:
            data = data - data.mean(dim=0, keepdim=True)
            data_lagged = data_lagged - data_lagged.mean(dim=0, keepdim=True)

        # Compute covariances
        c00 = data.t() @ data / (n_samples - 1)  # Instantaneous covariance
        c0t = data.t() @ data_lagged / (n_samples - 1)  # Time-lagged covariance
        ctt = data_lagged.t() @ data_lagged / (n_samples - 1)  # Time-lagged instantaneous covariance

        return c00, c0t, ctt

    def _sym_inverse(self, matrix: torch.Tensor, return_sqrt: bool = False) -> torch.Tensor:
        """
        Compute the inverse or inverse square root of a symmetric matrix.

        Parameters
        ----------
        matrix : torch.Tensor
            Symmetric matrix to invert
        return_sqrt : bool, default=False
            If True, return the inverse square root instead of the inverse

        Returns
        -------
        torch.Tensor
            Inverse or inverse square root of the matrix
        """
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

        # Handle small eigenvalues based on mode
        if self.mode == 'trunc':
            # Truncate small eigenvalues
            mask = eigenvalues > self.epsilon
            eigenvalues = eigenvalues[mask]
            eigenvectors = eigenvectors[:, mask]
        else:  # regularize
            # Add epsilon to eigenvalues for regularization
            eigenvalues = eigenvalues + self.epsilon

        # Compute inverse or inverse square root
        if return_sqrt:
            inv_eigenvalues = 1.0 / torch.sqrt(eigenvalues)
        else:
            inv_eigenvalues = 1.0 / eigenvalues

        # Reconstruct matrix
        inv_matrix = eigenvectors @ torch.diag(inv_eigenvalues) @ eigenvectors.t()

        return inv_matrix

    def _multi_dot(self, matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the product of multiple matrices efficiently.

        Parameters
        ----------
        matrices : List[torch.Tensor]
            List of matrices to multiply

        Returns
        -------
        torch.Tensor
            Product of all matrices
        """
        result = matrices[0]
        for matrix in matrices[1:]:
            result = result @ matrix
        return result
