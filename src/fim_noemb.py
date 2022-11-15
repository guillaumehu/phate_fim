import torch
import numpy as np


def torch_phate(X, kernel, bandwidth, t):

    """logarithm of the PHATE transition matrix."""
    dists = torch.norm(X[:, None] - X, dim=2, p="fro")

    def gaussian_kernel(x):
        return torch.exp(-(dists**2) / bandwidth)

    kernel = gaussian_kernel(dists)
    p = kernel / kernel.sum(axis=0)[:, None]
    pt = torch.matrix_power(p, t)
    log_p = torch.log(pt)
    return log_p


class FIM_noemb:
    """Fisher information matrix from a dataset. Using PHATE embedding with a Gaussian kernel."""

    def __init__(self, kernel="gaussian", bandwidth=10, t=10) -> None:
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.t = t

    def fit(self, X):

        """logarithm of the PHATE transition matrix, and computation of FIM"""
        n_obs, n_dim = X.shape

        # Jacobian
        self.log_p = torch_phate(
            X, kernel=self.kernel, bandwidth=self.bandwidth, t=self.t
        )
        fn = lambda x: torch_phate(
            x, kernel=self.kernel, bandwidth=self.bandwidth, t=self.t
        )
        J = torch.autograd.functional.jacobian(fn, X)

        jac = torch.empty((n_obs, n_obs, n_dim))
        for i in range(n_obs):
            jac[i] = J[i][i]

        # FIM
        prod = torch.ones((n_obs, n_dim, n_dim, n_obs))
        for i in range(n_dim):
            for j in range(i, n_dim):
                prod[:, i, j, :] = prod[:, j, i, :] = (
                    jac[:, :, i] * jac[:, :, j] * torch.exp(self.log_p)
                )

        fim = torch.sum(prod, dim=3)

        return fim

    def get_volume(self, X):
        fim = self.fit(X)
        V = np.sqrt(np.linalg.det(fim.detach().cpu()))
        return V
