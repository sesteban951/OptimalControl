
# standard imports
import numpy as np
from dataclasses import dataclass

# casadi import
import casadi as ca

##############################################################
# Generic NLP Solver
##############################################################

@dataclass
class NLPSolverParams:

    # timing parameters
    dt: float   # time step
    T: float    # time horizon

    # initial and final states
    x_init: np.ndarray  # initial state
    x_goal: np.ndarray  # goal state

@dataclass
class NLPResult:
    
    # solution arrays
    timespan: np.ndarray  # time trajectory
    X_opt: np.ndarray     # state trajectory
    U_opt: np.ndarray     # input trajectory

# generic NLP Solver class
class NLPSolver:

    # initialization
    def __init__(self, dynamics, params):

        # store the problem data
        self.dynamics = dynamics
        self.params   = params

        # build functions
        self.make_function()

    # build functions
    def make_function(self):

        # horiZon parameters
        self.dt = self.params.dt
        self.N = int(self.params.T / self.dt)
        
        # build state variables
        self.nx = self.dynamics.nx
        self.nu = self.dynamics.nu
        self.xk = ca.MX.sym('x', self.nx)
        self.uk = ca.MX.sym('u', self.nu)

        # build dynamics and cost function
        self.f = self.dynamics.f_disc
        self.cost = self.dynamics.cost

        # initialize and final states
        self.x_init = self.params.x_init
        self.x_goal = self.params.x_goal

    # solve the NLP
    def solve(self):

        # make the optimizer
        opti = ca.Opti()

        # horizon variable
        X = opti.variable(self.nx, self.N + 1)  # state trajectory
        U = opti.variable(self.nu, self.N)      # input trajectory

        # set the initial and goal state constraints
        opti.subject_to(X[:, 0]     == self.x_init)
        opti.subject_to(X[:, self.N] == self.x_goal)

        # system dynamics constraints at each time step
        for k in range(self.N):
            opti.subject_to(X[:, k + 1] == self.f(X[:, k], U[:, k]))

        # add input constraints if any
        if hasattr(self.dynamics, 'u_lb') and hasattr(self.dynamics, 'u_ub'):
            # input bounds
            u_lb = self.dynamics.u_lb
            u_ub = self.dynamics.u_ub

            # add to optimizer
            opti.subject_to(opti.bounded(u_lb, U, u_ub))

        # add state constraints if any
        if hasattr(self.dynamics, 'x_lb') and hasattr(self.dynamics, 'x_ub'):
            # state bounds
            x_lb = self.dynamics.x_lb
            x_ub = self.dynamics.x_ub

            # add to optimizer
            opti.subject_to(opti.bounded(x_lb, X, x_ub))

        # objective: minimize total cost
        J = 0
        for k in range(self.N):
            J += self.cost(X[:, k], U[:, k], self.x_goal)
        opti.minimize(J)

        # initial guess
        opti.set_initial(X, 0)
        opti.set_initial(U, 0)

        # solve with ipopt
        opti.solver('ipopt')
        sol = opti.solve()

        # extract the optimal trajectories and store
        X_opt = sol.value(X).T
        U_opt = sol.value(U).T
        timespan = np.linspace(0, self.params.T, self.N + 1)

        # store
        result = NLPResult(timespan=timespan,
                           X_opt=X_opt, 
                           U_opt=U_opt)
        
        return result
