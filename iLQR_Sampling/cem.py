##
#
# CEM sampling 
#
##

import torch
from cost import *
from pendulum import *
from dataclasses  import dataclass


# CEM parameters
@dataclass
class CEMParams:

    # integration steps
    N: int
    dt: float

    # sampling
    B: int         # number of rollouts
    n_elite: int   # top elite samples

    # distribution (full-covariance multivariate normal over the flattened
    # control sequence, dim M = (N-1) * m)
    mu0: float         # initial mean (scalar; broadcast to (M,))
    sigma0: float      # initial std (scalar; initial covariance = sigma0**2 * I_M)
    sigma_min: float   # minimum std (eigenvalue floor on covariance, as a std)
    sigma_max: float   # maximum std (eigenvalue ceiling on covariance, as a std)

    # iterations
    iter: int # number of iterations

# CEM Class
class CEM:

    def __init__(self, cem_params):
        
        # store the parameters
        self.params = cem_params

        # initialize the distribution
        self.initialize_distribution()


    # initialize the distribution: mu in R^M, sigma in R^{M x M}
    def initialize_distribution(self):
        N = self.params.N
        m = 1                     # control dimension
        M = m * (N - 1)           # flattened control-sequence dimension
        self.mu    = torch.full((M,), float(self.params.mu0))
        self.sigma = (float(self.params.sigma0) ** 2) * torch.eye(M)

    # sample from the distribution
    def sample_inputs(self):
        B = self.params.B
        N = self.params.N
        m = 1
        M = m * (N - 1)

        # cholesky of the (symmetrized) covariance + small ridge for numerical stability
        eps = 1e-6
        Sigma = 0.5 * (self.sigma + self.sigma.T) + eps * torch.eye(M)
        L = torch.linalg.cholesky(Sigma)                          # (M, M)

        # transform: U_flat = mu + Z @ L^T, so cov(U_flat) = Sigma
        Z = torch.randn(B, M)                                     # (B, M)
        U_flat = self.mu.unsqueeze(0) + Z @ L.T                   # (B, M)

        return U_flat.view(B, N - 1, m)                           # (B, N-1, m)

    # update the distribution from the elite samples (full covariance fit)
    def update_distribution(self, U_elite):
        K_elite = U_elite.shape[0]
        M = U_elite.shape[1] * U_elite.shape[2]

        # flatten the elite trajectories: (K_elite, N-1, m) -> (K_elite, M)
        U_flat = U_elite.reshape(K_elite, M)

        # mean
        mu_new = U_flat.mean(dim=0)                               # (M,)

        # centered covariance
        U_centered = U_flat - mu_new.unsqueeze(0)
        sigma_new  = (U_centered.T @ U_centered) / (K_elite - 1)  # (M, M)

        # symmetrize for numerical stability
        sigma_new = 0.5 * (sigma_new + sigma_new.T)

        # eigenvalue clamp: floor / ceiling on covariance (as variances)
        eigvals, eigvecs = torch.linalg.eigh(sigma_new)
        eigvals = eigvals.clamp(self.params.sigma_min ** 2,
                                self.params.sigma_max ** 2)
        sigma_new = eigvecs @ torch.diag(eigvals) @ eigvecs.T

        # store
        self.mu    = mu_new
        self.sigma = sigma_new
