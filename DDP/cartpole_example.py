
# standard imports
import numpy as np
import sympy as sym  # for symbolic math
from matplotlib import pyplot as plt

# custom imports
from ddp import DDP, DDPParams

##############################################################
# Cartpole for DDP
##############################################################

# system dynamics
class Cartpole:
    """
    CartPole Dynamics (underactuated robotics Ch 2.2)
        (m_c + m_p) ẍ + m_p l cos(θ) θ̈ - m_p l sin(θ) θ̇ ² + bp θ̇ = u
        ẍ cos(θ) + l θ̈ + g sin(θ) + bp ẋ = 0

        x = [cosθ, sinθ, cart pos, θdot, xdot]
    """

    # initialize
    def __init__(self):
        
        # state and input dimensions
        self.nx = 5    # overloaded to 5 for cos(θ), sin(θ), cart pos, θ̇, ẋ
        self.nu = 1    # torque

        # input limits
        self.u_lims = (-2, 2)

        # system parameters
        self.L = 1.0    # length (m)
        self.mp = 1.0   # mass of the pole (kg)
        self.mc = 1.0   # mass of the cart (kg)
        self.bp = 0.1   # damping for pole
        self.bc = 0.1   # damping for cart
        self.g = 9.81   # gravity (m/s^2)

        # projection onto unit circle epsilon
        self.eps = 1e-9

        # cost weights
        self.Q  = np.diag([10.0,  10.0, 8.0, 8.0, 1.0])  # state cost weights
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
        p = x[2]  # cart position
        w = x[3]  # θ̇
        v = x[4]  # cart velocity

        # continuous dynamics
        c_dot = -s * w
        s_dot =  c * w
        w_dot = (
            (-u * c - self.mp * self.L * w**2 * s * c - (self.mc + self.mp) * self.g * s + self.bc * v * c - self.bp * w) 
            / (self.L * (self.mc + self.mp * s**2))
        )
        v_dot = (
            (u + self.mp * s * (self.L * w**2 + self.g * c) - self.bc * v) 
            / (self.mc + self.mp * s**2)
        )

        xdot = sym.Matrix([c_dot, s_dot, v, w_dot, v_dot])

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

    # create DDP params
    ddp_params = DDPParams(
        dt=0.04,
        max_iters=250,
        tol=1e-6,
        alphas=[1.0, 0.75, 0.5, 0.25, 0.1, 0.01],
        use_dyn_hessians=False
    )

    # create DDP optimizer
    ddp = DDP(system=Cartpole(), 
              params=ddp_params)

    # initial state
    theta0 = 0.0
    x0 = np.array([np.cos(theta0), 
                   np.sin(theta0), 
                   0.0, 
                   0.0,
                   0.0]).reshape(5,1)  
    theta_des = np.pi
    xdes = np.array([np.cos(theta_des), 
                     np.sin(theta_des),
                     0.0,
                     0.0, 
                     0.0]).reshape(5,1)  # desired state

    # total time
    T = 7.0

    # run DDP optimization
    X, U = ddp.optimize(x0, xdes, T)

    # get angle from sin and cos
    angles = np.arctan2(X[:,1], X[:,0])
    X = np.hstack((angles.reshape(-1,1), X[:,2:]))  # replace first two columns with angle
    
    # compute the time span
    time = np.arange(0, T, ddp.dt)

    # save the results into a CSV file
    times_filename = "./DDP/results/cartpole_time.csv"
    states_filename = "./DDP/results/cartpole_state.csv"
    inputs_filename = "./DDP/results/cartpole_input.csv"
    np.savetxt(times_filename, time, delimiter=",")
    np.savetxt(states_filename, X, delimiter=",")
    np.savetxt(inputs_filename, U, delimiter=",")
    print(f"Saved time to {times_filename}")
    print(f"Saved states to {states_filename}") 
    print(f"Saved inputs to {inputs_filename}")

    # plot states and inputs
    plt.figure(figsize=(12,8))
    plt.subplot(3,1,1)
    plt.plot(time, X[:,0], label="Angle (rad)")
    plt.ylabel("Angle (rad)")
    plt.grid()
    plt.subplot(3,1,2)
    plt.plot(time, X[:,2], label="Cart Position (m)", color='orange')
    plt.ylabel("Cart Position (m)")
    plt.grid()
    plt.subplot(3,1,3)
    plt.plot(time[:-1], U[:,0], label="Input (N)", color='green')
    plt.ylabel("Input (N)")
    plt.xlabel("Time (s)")
    plt.grid()
    plt.suptitle("CartPole DDP Results")
    plt.show()