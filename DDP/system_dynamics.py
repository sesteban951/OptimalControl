
# standard imports
import numpy as np
import sympy as sym  # for symbolic math

##############################################################
# Inverted Pendulum for DDP
##############################################################

# system dynamics
class Pendulum:
    """
    x = [cos(θ); sin(θ); θ̇]
    ẋ = [-sin(θ) * θ̇  ;
          cos(θ) * θ̇  ;
         -(g/l) * sin(θ) - (b/(m*l^2)) * θ̇ + (1/(m*l^2)) * u
    ]
    """

    # initialize
    def __init__(self):
        
        # state and input dimensions
        self.nx = 3    # overloaded to 3 for cos(θ), sin(θ), θ̇
        self.nu = 1    # torque 

        # input limits
        self.u_lims = (-2, 2)

        # system parameters
        self.L = 1.0      # length (m)
        self.m = 1.0      # mass (kg)
        self.g = 9.81     # gravity (m/s^2)
        self.b = 0.1      # damping coefficient

        # projection onto unit circle epsilon
        self.eps = 1e-9

        # cost weights
        self.Q = np.diag([10.0, 10.0, 1.0])      # state cost weights
        self.R = np.diag([0.1])                  # input cost weights
        self.Qf = np.diag([100.0, 100.0, 10.0])  # terminal state cost weights

    #######################################################
    # DYNAMICS
    #######################################################

    # continous-time dynamics
    def f_c(self, x, u):

        # scalar torque even if u is a (1x1) Matrix
        if isinstance(u, sym.MatrixBase):
            u = u[0]

        # unpack state
        c = x[0]  # cos(θ)
        s = x[1]  # sin(θ)
        w = x[2]  # θ̇

        # continuous dynamics
        c_dot = -s * w
        s_dot =  c * w
        w_dot = -(self.g / self.L) * s - (self.b / (self.m * self.L**2)) * w + (1 / (self.m * self.L**2)) * u
        xdot = sym.Matrix([[c_dot], 
                           [s_dot], 
                           [w_dot]])

        return xdot
    
    # discrete-time dynamics (using RK4)
    def f_d(self, x, u, dt):

        # RK4 integration
        k1 = self.f_c(x, u)
        k2 = self.f_c(x + 0.5 * dt * k1, u)
        k3 = self.f_c(x + 0.5 * dt * k2, u)
        k4 = self.f_c(x + dt * k3, u)

        # take the step
        x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # normalize cos and sin to constrain to unit circle
        c = x_next[0]
        s = x_next[1]
        r = sym.sqrt(c**2 + s**2 + self.eps)  # avoid division by zero
        x_next[0] = c / r
        x_next[1] = s / r

        return sym.Matrix(x_next)
    
    #######################################################
    # COST FUNCTIONS
    #######################################################

    # stage cost
    def l(self, x, u, xdes):

        # compute the state error
        e = x - xdes

        # total cost
        c = 0.5 * e.T @ self.Q @ e  + 0.5 * u.T @ self.R @ u

        return c

    # terminal cost
    def lf(self, x, xdes):

        # compute the state error
        e = x - xdes

        # total terminal cost
        c = 0.5 * e.T @ self.Qf @ e

        return c

#######################################################
# Test
#######################################################

# example usage
if __name__ == "__main__":

    # create an instance of the dynamics
    pendulum = Pendulum()

    # define symbolic state and input
    c, s, w = sym.symbols('c s w')  # cos(θ), sin(θ), θ̇
    u = sym.symbols('u')             # torque
    x = sym.Matrix([[c], 
                    [s],
                    [w]])

    # time step
    dt = 0.01

    # compute next state
    x_next = pendulum.f_d(x, u, dt)

    # compute Jacobians
    A = x_next.jacobian(x)

    print("Next state x_next:")
    sym.pprint(x_next)
    print("\nJacobian A = ∂f/∂x:")
    sym.pprint(A)