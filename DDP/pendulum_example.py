
# standard imports
import numpy as np
import sympy as sym  # for symbolic math
from matplotlib import pyplot as plt

# custom imports
from ddp import DDP, DDPParams

##############################################################
# Inverted Pendulum for DDP
##############################################################

# system dynamics
class Pendulum:
    """
    Inverted Pendulum Dynamics:
        x = [cos(θ); sin(θ); θ̇]
        ẋ = [-sin(θ) * θ̇  ;
              cos(θ) * θ̇  ;
             -(g/l) * sin(θ) - (b/(m*l^2)) * θ̇ + (1/(m*l^2)) * u]
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
        self.Q  = np.array([[10.0, 0.0,  0.0],
                            [0.0,  10.0, 0.0],
                            [0.0,  0.0,  2.0]])  # state cost weights
        self.R  = np.diag([0.01])                 # input cost weights
        self.Qf = 10 * self.Q                    # terminal state cost weights

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
    
    # discrete-time dynamics (using Euler integration)
    def f_d (self, x, u, dt):
        
        # Euler integration
        x_next = x + dt * self.f_c(x, u)

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
        c = 0.5 * e.T @ self.Q @ e  + 0.5 * u.T @ self.R @ u

        return c
        
    # terminal cost
    def lf(self, x, xdes):

        # compute the state error
        e = x - xdes
        c = 0.5 * e.T @ self.Qf @ e

        return c

#######################################################
# EXAMPLE USAGE
#######################################################

# example usage
if __name__ == "__main__":

    # create an instance of the dynamics
    pendulum = Pendulum()

    # create DDP params
    ddp_params = DDPParams(
        dt=0.04,
        max_iters=250,
        tol=1e-6,
        alphas=[1.0, 0.75, 0.5, 0.25, 0.1, 0.01],
        use_dyn_hessians=False
    )

    # create DDP optimizer
    ddp = DDP(system=Pendulum(), 
              params=ddp_params)
    
        # create DDP params
    ddp_params = DDPParams(
        dt=0.04,
        max_iters=250,
        tol=1e-6,
        alphas=[1.0, 0.75, 0.5, 0.25, 0.1, 0.01],
        use_dyn_hessians=False
    )

    # create DDP optimizer
    ddp = DDP(system=Pendulum(), 
              params=ddp_params)

    # initial state
    theta0 = 0.0
    x0 = np.array([np.cos(theta0), 
                   np.sin(theta0), 
                   0.0]).reshape(3,1)  # [theta, theta_dot]
    theta_des = np.pi
    xdes = np.array([np.cos(theta_des), 
                     np.sin(theta_des), 
                     0.0]).reshape(3,1)  # desired state

    # total time
    T = 5.0

    # run DDP optimization
    X, U = ddp.optimize(x0, xdes, T)

    # get angle from sin and cos
    angles = np.arctan2(X[:,1], X[:,0])
    X = np.hstack((angles.reshape(-1,1), X[:,2].reshape(-1,1)))  # [theta, theta_dot]

    # compute the time span
    time = np.arange(0, T, ddp.dt)

    # plot the results
    plt.figure(figsize=(10,6))
    plt.subplot(3,1,1)
    plt.plot(time, X[:,0], label='Theta (rad)')
    plt.axhline(y=theta_des, color='k', linestyle='--', label='Desired Theta')
    plt.axhline(y=-theta_des, color='k', linestyle='--', label='Desired Theta')
    plt.ylabel('Theta (rad)')
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(time, X[:,1], label='Theta dot (rad/s)')
    plt.axhline(y=xdes[2], color='k', linestyle='--')
    plt.ylabel('Theta dot (rad/s)')
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(time[:-1], U, label='Control Input (u)')
    plt.axhline(y=0, color='k', linestyle='--', label='Desired Control Input')
    plt.ylabel('Control Input (u)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.tight_layout()
    plt.show()