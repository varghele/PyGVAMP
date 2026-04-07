import torch
import torch.nn as nn
import torch.nn.functional as F


class ReversibleVAMPScore(nn.Module):
    """
    Reversible VAMP score module implementing likelihood-based training
    with detailed balance constraints.

    Learns a transition matrix K that satisfies detailed balance by construction:
        K_ij = S_ij * u_j / sum_k(S_ik * u_k)
    where S is a symmetric non-negative rate matrix and u is the stationary distribution.

    Parameters
    ----------
    n_states : int
        Number of metastable states
    epsilon : float, default=1e-6
        Numerical stability constant
    """

    def __init__(self, n_states: int, epsilon: float = 1e-6):
        super(ReversibleVAMPScore, self).__init__()

        self.n_states = n_states
        self.epsilon = epsilon

        # Learnable parameters
        self.log_stationary = nn.Parameter(torch.zeros(n_states))
        self.rate_matrix_weights = nn.Parameter(torch.zeros(n_states, n_states))

    def get_stationary_distribution(self) -> torch.Tensor:
        """
        Compute the stationary distribution from unconstrained log weights.

        Returns
        -------
        torch.Tensor
            Stationary distribution of shape (n_states,), positive and sums to 1
        """
        return torch.softmax(self.log_stationary, dim=0)

    def get_rate_matrix(self) -> torch.Tensor:
        """
        Compute the symmetric non-negative rate matrix.

        Returns
        -------
        torch.Tensor
            Rate matrix of shape (n_states, n_states), symmetric and non-negative
        """
        W = (self.rate_matrix_weights + self.rate_matrix_weights.T) / 2
        S = F.softplus(W)
        return S

    def get_transition_matrix(self) -> torch.Tensor:
        """
        Compute the transition matrix satisfying detailed balance.

        Construction:
            K[i,j] = S[i,j] * u[j]  for i != j  (off-diagonal)
            K[i,i] = 1 - sum_{j!=i} K[i,j]       (diagonal, ensures row-stochasticity)

        S is scaled so that off-diagonal row sums don't exceed 1, ensuring
        non-negative diagonal entries.

        This guarantees u[i]*K[i,j] = u[j]*K[j,i] (detailed balance) because
        S is symmetric: u[i]*S[i,j]*u[j] = u[j]*S[j,i]*u[i].

        Returns
        -------
        torch.Tensor
            Row-stochastic transition matrix of shape (n_states, n_states)
        """
        u = self.get_stationary_distribution()
        S = self.get_rate_matrix()

        # Off-diagonal: K[i,j] = S[i,j] * u[j]
        K_offdiag = S * u.unsqueeze(0)

        # Zero out diagonal
        mask = 1.0 - torch.eye(self.n_states, device=S.device)
        K_offdiag = K_offdiag * mask

        # Scale S so that max off-diagonal row sum <= 1 - epsilon
        # This ensures non-negative diagonal entries
        row_sums = K_offdiag.sum(dim=1)
        max_row_sum = row_sums.max()
        if max_row_sum > 1.0 - self.epsilon:
            scale = (1.0 - self.epsilon) / (max_row_sum + self.epsilon)
            K_offdiag = K_offdiag * scale

        # Set diagonal to ensure rows sum to 1
        diag = 1.0 - K_offdiag.sum(dim=1)
        K = K_offdiag + torch.diag(diag)

        return K

    def forward(self, chi_t0: torch.Tensor, chi_t1: torch.Tensor) -> torch.Tensor:
        """
        Compute the negative log-likelihood loss.

        Parameters
        ----------
        chi_t0 : torch.Tensor
            Softmax probabilities at time t, shape (batch, n_states)
        chi_t1 : torch.Tensor
            Softmax probabilities at time t+tau, shape (batch, n_states)

        Returns
        -------
        torch.Tensor
            Scalar negative log-likelihood loss
        """
        K = self.get_transition_matrix()

        # Transition probabilities: p_i = sum_j chi_t0[i,j] * K[j,k] * chi_t1[i,k]
        p = (chi_t0 @ K) * chi_t1
        p = p.sum(dim=1)  # shape (batch,)

        # Negative log-likelihood
        return -torch.log(torch.clamp(p, min=self.epsilon)).mean()

    def loss(self, chi_t0: torch.Tensor, chi_t1: torch.Tensor) -> torch.Tensor:
        """
        Compute loss (API consistency with VAMPScore).

        Parameters
        ----------
        chi_t0 : torch.Tensor
            Softmax probabilities at time t
        chi_t1 : torch.Tensor
            Softmax probabilities at time t+tau

        Returns
        -------
        torch.Tensor
            Negative log-likelihood loss
        """
        return self.forward(chi_t0, chi_t1)
