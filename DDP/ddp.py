# standard library
import numpy as np   
from numpy.typing import ArrayLike
from dataclasses import dataclass

# symbolic math
import sympy as sym  

# custom import 
from system_dynamics import Pendulum

##############################################################
# Differential Dynamic Programming (DDP) Class
##############################################################

# params for DDP
@dataclass
class DDPParams:
    
    # DDP parameters
    max_iters: int = 100        # maximum DDP iterations
    tol: float = 1e-6           # convergence tolerance
    alphas: ArrayLike = [1.0],  # line search parameters for backtracking
    use_hessians: bool = True   # use 2nd order dynamics derivatives

# main DDP class
class DDP:

    # initialize
    def __init__(self,
                 system,
                 x0, xdes,
                 T, dt
                 ):

        # dynamics
        self.nx = system.nx          # state dimension
        self.nu = system.nu          # input dimension
        self.u_lims = system.u_lims  # input limits
        self.f_ = system.f_d        # discrete dynamics function

        # cost functions
        self.l_  = system.l              # running cost
        self.lf_ = system.lf            # terminal cost
        
        # initial and desired state state
        self.x0 = x0
        self.xdes = xdes

        # time horizon
        self.T = T
        self.dt = dt

        # make the functions
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
    def optimize(self):

        pass # TODO: implement DDP optimization loop


# example usage
if __name__ == "__main__":

    # create DDP params
    ddp_params = DDPParams()

    # create DDP optimizer
    ddp = DDP(Pendulum(), 
              x0=np.zeros(4), 
              xdes=np.zeros(4), 
              T=5.0, dt=0.04)
