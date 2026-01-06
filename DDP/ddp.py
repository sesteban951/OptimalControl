# standard library
import numpy as np   
from numpy.typing import ArrayLike
from dataclasses import dataclass
import matplotlib.pyplot as plt

# symbolic math
import sympy as sym  


##############################################################
# Differential Dynamic Programming (DDP) Class
##############################################################

# params for DDP
@dataclass
class DDPParams:

    # time parameters
    dt: float = 0.04               # time step
    
    # DDP parameters
    max_iters: int = 100           # maximum DDP iterations
    tol: float = 1e-6              # convergence tolerance
    alphas: ArrayLike = [1.0],     # line search parameters for backtracking
    use_dyn_hessians: bool = False # use 2nd order dynamics derivatives

# main DDP class
class DDP:

    # initialize
    def __init__(self, system, params):

        # DDP optimization parameters
        self.params = params

        # dynamics
        self.nx = system.nx          # state dimension
        self.nu = system.nu          # input dimension
        self.u_lims = system.u_lims  # input limits
        self.f_ = system.f_d        # discrete dynamics function

        # cost functions
        self.l_  = system.l              # running cost
        self.lf_ = system.lf            # terminal cost

        # time parameters
        self.dt = params.dt

        # make all functions and derivatives
        self.make_functions()

    # make the dynamics functions and their gradients
    def make_functions(self):

        print("Making functions and derivatives...")

        # state symbols
        x = sym.symbols("x:{:}".format(self.nx)) # (x0, x1, ..., x_{nx-1})
        x = sym.Matrix([xi for xi in x])         # nx x 1 SymPy Matrix

        # control input symbols
        u = sym.symbols("u:{:}".format(self.nu)) # (u0, u1, ..., u_{nu-1})
        u = sym.Matrix([ui for ui in u])         # nu x 1 SymPy Matrix

        # desired state symbols
        xdes = sym.symbols("xdes:{:}".format(self.nx)) # (xdes0, xdes1, ..., xdes_{nx-1})
        xdes = sym.Matrix([xdi for xdi in xdes])       # nx x 1 SymPy Matrix

        # dynamics function
        print("\tMaking dynamics function...")
        self.f = sym.lambdify((x, u), self.f_(x, u, self.dt))  # x_next = f(x, u, dt)

        # 1st order derivatives
        print("\tMaking 1st order dynamics derivatives...")
        f_x = self.f_(x, u, self.dt).jacobian(x)  # Jx = ∂f/∂x  (nx × nx)
        f_u = self.f_(x, u, self.dt).jacobian(u)  # Ju = ∂f/∂u  (nx × nu)
        self.f_x = sym.lambdify((x, u), f_x)      # ∂f/∂x  (nx × nx)
        self.f_u = sym.lambdify((x, u), f_u)      # ∂f/∂u  (nx × nu)

        # 2nd order derivatives
        if self.params.use_dyn_hessians == True:
            print("\tMaking 2nd order dynamics derivatives...")
            self.f_xx = sym.lambdify((x, u), [f_x.row(i).jacobian(x) for i in range(self.nx)])  # ∂²f/∂x²   (nx × nx × nx) tensor
            self.f_ux = sym.lambdify((x, u), [f_u.row(i).jacobian(x) for i in range(self.nx)])  # ∂²f/∂u∂x  (nx × nu × nx) tensor
            self.f_uu = sym.lambdify((x, u), [f_u.row(i).jacobian(u) for i in range(self.nx)])  # ∂²f/∂u²   (nx × nu × nu) tensor

        # cost functions
        print("\tMaking cost functions...")
        self.l  = sym.lambdify((x, u, xdes), self.l_(x, u, xdes))  # running cost
        self.lf = sym.lambdify((x, xdes), self.lf_(x, xdes))       # terminal cost

        # 1st order derivatives
        print("\tMaking 1st order cost derivatives...")
        self.l_x  = sym.lambdify((x, u, xdes), self.l_(x, u, xdes).jacobian(x))  # ∂l/∂x  (1 × nx)
        self.l_u  = sym.lambdify((x, u, xdes), self.l_(x, u, xdes).jacobian(u))  # ∂l/∂u  (1 × nu)
        self.lf_x = sym.lambdify((x, xdes), self.lf_(x, xdes).jacobian(x))       # ∂lf/∂x (1 × nx)

        # 2nd order derivatives
        print("\tMaking 2nd order cost derivatives...")
        self.l_xx = sym.lambdify((x, u, xdes), self.l_(x, u, xdes).jacobian(x).jacobian(x)) # ∂²l/∂x²  (nx × nx)
        self.l_uu = sym.lambdify((x, u, xdes), self.l_(x, u, xdes).jacobian(u).jacobian(u)) # ∂²l/∂u²  (nu × nu)
        self.l_ux = sym.lambdify((x, u, xdes), self.l_(x, u, xdes).jacobian(u).jacobian(x)) # ∂²l/∂u∂x (nu × nx)
        self.lf_xx = sym.lambdify((x, xdes), self.lf_(x, xdes).jacobian(x).jacobian(x))     # ∂²lf/∂x² (nx × nx)

    # do main DDP optimization
    def optimize(self, x0, xdes, T):

        # compute the trajectory length 
        N = int(T // self.dt)

        # generate a random control sequence from normal distribution
        U = np.random.randn(N, self.nu) * 0.1

        # convenience function to compute cost
        def J(X, U):
            c_total = 0.0
            for k in range(len(U)):
                c_total += self.l(X[k], U[k], xdes)
            c_total += self.lf(X[-1], xdes)

            return c_total
        
        # rollout the initial trajectory
        X = np.zeros((N+1, self.nx))
        X[0] = x0.flatten()
        for k in range(N):
            X[k+1] = self.f(X[k], U[k]).flatten() # x_next = f(x, u)  (discrete dynamics)

        # compute initial cost
        J_curr = J(X, U)

        # flag to stop optimization
        done = False

        # DDP main loop
        print("Starting DDP optimization...")
        for i in range(self.params.max_iters):  

            print(f"Iteration {i+1}")

            # backwards pass
            V_x  = self.lf_x(X[-1],  xdes).flatten()  # Vx = ∂V/∂x at final time
            V_xx = self.lf_xx(X[-1], xdes)            # Vxx = ∂²V/∂x² at final time

            # create arrays for Q derivatives
            Qu_list =  np.zeros((N, self.nu))
            Quu_list = np.zeros((N, self.nu, self.nu))
            Qux_list = np.zeros((N, self.nu, self.nx))

            # ----------------------------------------
            # BACKWARDS PASS
            # ----------------------------------------
            for k in reversed(range(N)):
                
                # get current state and input
                x_k = X[k]
                u_k = U[k]

                # dynamics approximation at (x_k, u_k)
                f_x  = self.f_x(x_k, u_k).reshape(self.nx, self.nx)
                f_u  = self.f_u(x_k, u_k).reshape(self.nx, self.nu)

                if self.params.use_dyn_hessians == True:
                    f_xx = np.stack([
                        np.asarray(H, dtype=float).reshape(self.nx, self.nx)
                        for H in self.f_xx(x_k, u_k)
                    ], axis=0)                           # (nx, nx, nx)
                    f_ux = np.stack([
                        np.asarray(H, dtype=float).reshape(self.nu, self.nx)
                        for H in self.f_ux(x_k, u_k)
                    ], axis=0)                           # (nx, nu, nx)
                    f_uu = np.stack([
                        np.asarray(H, dtype=float).reshape(self.nu, self.nu)
                        for H in self.f_uu(x_k, u_k)
                    ], axis=0)                           # (nx, nu, nu)

                # cost approximation at (x_k, u_k)
                l_x  = self.l_x(x_k, u_k, xdes).flatten()      
                l_u  = self.l_u(x_k, u_k, xdes).flatten()
                l_xx = self.l_xx(x_k, u_k, xdes)
                l_ux = self.l_ux(x_k, u_k, xdes)
                l_uu = self.l_uu(x_k, u_k, xdes)

                # compute Q function derivatives
                Q_x  = l_x + f_x.T @ V_x
                Q_u  = l_u + f_u.T @ V_x
                Q_xx = l_xx + f_x.T @ V_xx @ f_x
                Q_ux = l_ux + f_u.T @ V_xx @ f_x
                Q_uu = l_uu + f_u.T @ V_xx @ f_u

                if self.params.use_dyn_hessians == True:
                    Q_xx += np.tensordot(V_x, f_xx, axes=(0,0))
                    Q_ux += np.tensordot(V_x, f_ux, axes=(0,0))
                    Q_uu += np.tensordot(V_x, f_uu, axes=(0,0))

                # store Q derivatives
                Qu_list[k]   = Q_u
                Quu_list[k]  = Q_uu
                Qux_list[k]  = Q_ux

                # compute optimal control law
                Quu_inv = np.linalg.pinv(Q_uu)  # pseudo-inverse for numerical stability
                V_x =  Q_x  - Q_ux.T @ Quu_inv @ Q_u
                V_xx = Q_xx - Q_ux.T @ Quu_inv @ Q_ux

            # ----------------------------------------
            # FORWARDS PASS w/ BACKTRACKING
            # ----------------------------------------
            for i, alpha in enumerate(self.params.alphas):

                # initialize optimal trajectory
                X_star = np.zeros_like(X)
                U_star = np.zeros_like(U)
                X_star[0] = x0.flatten()

                # compute the new trajectory with the updated control law
                for k in range(N):

                    # current Q derivatives
                    Q_u  = Qu_list[k]
                    Q_ux = Qux_list[k]
                    Q_uu = Quu_list[k]

                    # compute error
                    e = X_star[k] - X[k]
                    U_star[k] = U[k] - np.linalg.pinv(Q_uu) @ (alpha * Q_u + Q_ux @ e)
                    X_star[k+1] = self.f(X_star[k], U_star[k]).flatten()

                # update cost  to see if there are improvements
                J_new = J(X_star, U_star)
                if J_new < J_curr:
                    X = X_star
                    U = U_star
                    break
                
                # if no cost improvement found
                if alpha == self.params.alphas[-1]:
                    print("No cost improvement found in line search.")
                    done = True
            
            # check for convergence at the end of the optimization cycle
            J_diff = abs(J_curr - J_new)
            if done or J_diff < self.params.tol:
                break

            # update current cost
            J_curr = J_new

        return X, U
