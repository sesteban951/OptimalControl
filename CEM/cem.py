import torch 
import numpy as np
from dataclasses  import dataclass
import os

from CEM.dynamics import Cartpole, DoubleIntegrator

##############################################################

# CEM parameters
@dataclass
class CEMParams:

    # integration steps
    N: int
    dt: float

    # sampling
    K: int         # number of rollouts
    N_elite: int   # top elite samples

    # distribution (normal)
    mu0: torch.Tensor        # inital mean
    sigma0: torch.Tensor     # inital covariance
    sigma_min: float   # minimum covariance
    sigma_max: float   # maximum cov

    # iterations
    iter: int # number of iterations

# Cross Entropy Method
class CEM():

    def __init__(self, params: CEMParams,
                       dynamics):

        # copy the dynamics
        self.dyn = dynamics

        # copy the parameters
        self.params = params

        # some checks
        assert self.params.K >= self.params.N_elite

        # initialize distribution
        self.mu = self.params.mu0
        self.sigma = self.params.sigma0

    # sample input sequnce
    def sample_inputs(self):

        # get the sizes
        K = self.params.K
        N = self.params.N
        nu = self.dyn.nu
        M = nu * N

        # faltten mu
        mu_ = self.mu
        if mu_.ndim == 2:
            mu_ = mu_.squeeze(1)  # (nx * N, 1) -> (nx * N, )

        # sample from standard normal
        U_std = torch.randn((K, M))

        # compute the cholesky facotrization fo sigma, Sigma = L * L^T
        eps = 1e-6
        I = torch.eye(M)
        Sigma = 0.5 * (self.sigma + self.sigma.T) + I * eps
        L = torch.linalg.cholesky(Sigma)  

        # transform to actual distirbution
        # (K, M) = (1, M) + (K,M) @ (M,M)   (broadcast summing)
        U_flat = mu_.unsqueeze(0) + U_std @L.T  # (K, M)
        U = U_flat.view(K, N, nu)                         # (K, N, nu)

        return U
    
    # forward propagate dynamics
    def forward_prop(self, x0, U):

        # preallocate solutions
        X = torch.empty(self.params.K, 
                        self.params.N+1, 
                        self.dyn.nx)  # shape(K, N+1, nx)

        # allocate the intial state
        X[:, 0, :] = x0

        # integrate given different inputs
        xk = x0
        for k in range(self.params.N):

            # get the current input vector
            uk = U[:, k, :]  # shape (K, nu)

            # forward prop
            xk = self.dyn.f_disc(xk, uk, self.params.dt) # shape (K, nx)

            # append to the trajecotries
            X[:, k+1, :] = xk

        return X
    
    # evaulte the trajecotries
    def eval_traj(self, X, U):

        # forward prop the trajectories
        J = self.dyn.cost(X, U, self.params.dt)  # shape (K, )

        # order the top N_elite trajectories
        J_elite, idx_elite = torch.topk(J, 
                                        k=self.params.N_elite,
                                        largest=False, 
                                        sorted = True) # shape(K, )
        X_elite = X[idx_elite] # shape (K, N+1, nx)
        U_elite = U[idx_elite] # shape (K, N)

        return X_elite, U_elite, J_elite
    
    # update the distribution
    def update_distirbution(self, U_elite):

        # extract dimensions
        K_elite, N_elite, nu = U_elite.shape
        M = nu * N_elite

        # flatten the elite trajectory (K_elite, N, nu) -> (K_elite, nu * N)
        U_flat = U_elite.reshape(K_elite, M)

        # compute the updated mean
        mu_ = U_flat.mean(dim=0)   # shape ()

        # center the trajectories about the mean
        U_centered = U_flat - mu_.unsqueeze(0)

        # compute covariance
        sigma_ = (U_centered.T @ U_centered) / (K_elite - 1)
        
        # update the distribution
        self.mu = mu_
        self.sigma = sigma_

    # perform the cross entropy method
    def cem(self, x0):

        # iterate
        for i in range(self.params.iter):

            print("iteration {}".format(i+1))
            
            # sample from the distribution
            U_traj = self.sample_inputs()

            # forward prop the under the inputs
            X_traj = self.forward_prop(x0, U_traj)

            # find the elite group
            X_elite, U_elite, J_elite = self.eval_traj(X_traj, U_traj)

            print(J_elite[0].item())

            # update the distribution
            self.update_distirbution(U_elite)

        # return the best 
        X_star = X_elite[0, ...] # shape (N+1, nx)
        U_star = U_elite[0, ...] # shape (N  , nu)

        return X_star, U_star

##############################################################

# example usage
if __name__ == "__main__":

    # put on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize the class
    # dyn = Cartpole()
    dyn = DoubleIntegrator()

    # settings
    T = 5.0
    dt = 0.025
    N = int(np.round(T/dt))

    # initial mean and covariance
    mu0 = torch.zeros(dyn.nu * N, 1)      # shape (nu * N, 1)
    sigma_diags = torch.ones(dyn.nu * N)  # shape (nu * N, )
    sigma0 = torch.diag(sigma_diags)      # shape (nu * N, nu * N)

    # parameters
    cem_params = CEMParams(
        N=N,
        dt=dt,
        K=4096,
        N_elite=100,
        mu0=mu0,
        sigma0=sigma0,
        sigma_min=0.01,
        sigma_max=10.0,
        iter=500
    )
    cem = CEM(cem_params, dyn)

    # sample initial condition
    x0_single = torch.tensor([0., 0., 0., 0.]) # shape (4, )
    x0 = x0_single.unsqueeze_(0).repeat(cem_params.K, 1)  # shape (K, nx)

    # run CEM
    X_star, U_star = cem.cem(x0)

    # compute a time trajectory
    time = np.linspace(0, N*dt, N+1)

    # save the results
    save_dir = "./CEM/results/"
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    np.savetxt(os.path.join(save_dir, "time.csv"), time, delimiter=",")
    np.savetxt(os.path.join(save_dir, "state.csv"), X_star.numpy(), delimiter=",")
    np.savetxt(os.path.join(save_dir, "input.csv"), U_star.numpy(), delimiter=",")
