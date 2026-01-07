
# standard imports
import numpy as np
import matplotlib.pyplot as plt

# casadi import
import casadi as ca

# custom imports
from nlp import NLPSolver, NLPSolverParams, NLPResult

##############################################################

# Pendulm for NLP example
class Cartpole:

    def __init__(self, dt):

        # state dimension
        self.nx = 4  # [theta, pos, theta_dot, pos_dot]

        # input dimension
        self.nu = 1  # [torque]

        # state limits
        self.x_lb = np.array([-np.inf, -3.0, -np.inf, -np.inf]).reshape(self.nx,1)  # lower state limits
        self.x_ub = np.array([ np.inf,  3.0,  np.inf,  np.inf]).reshape(self.nx,1)  # upper state limits

        # input limits
        self.u_lb = np.array([-3.0]).reshape(self.nu,1)  # lower input limits
        self.u_ub = np.array([ 3.0]).reshape(self.nu,1)  # upper input limits

        # time step
        self.dt = dt

    # cartpole continuous dynamics
    def f_cont(self, x, u):

        # cartpole parameters
        mc = 1.0
        mp = 0.2
        g = 9.81
        l = 0.5

        # unpack state
        th = x[0]     # pole angle
        th_dot = x[2] # pole angular velocity
        p_dot = x[3]  # cart velocity
    
        # useful quantities
        sin_th = ca.sin(th)
        cos_th = ca.cos(th)
        denom = mc + mp * sin_th**2

        # dynamics equations
        th_ddot = (-u[0] * cos_th- mp * l * th_dot**2 * cos_th * sin_th - (mc + mp) * g * sin_th) / (l * denom)
        p_ddot  = (u[0] + mp * sin_th * (l * th_dot**2 + g * cos_th)) / denom

        # dynamics
        xdot = ca.vertcat(th_dot, p_dot, th_ddot, p_ddot)

        return xdot
    
    # discrete dynamics
    def f_disc(self, xk, uk):

        # euler integration
        xk_next = xk + self.dt * self.f_cont(xk, uk)

        return xk_next
    
    # cost function
    def cost(self, x, u, x_goal):

        # weights
        w_th    = 20.0
        w_thdot = 2.0
        w_pos   = 10.0
        w_vel   = 1.0
        Q = ca.diag(ca.vertcat(w_th, w_pos, w_thdot, w_vel))

        w_u     = 1.0
        R = ca.diag(ca.vertcat(w_u))

        # compute error terms
        th_err  = 1 - ca.cos(x[0] - x_goal[0])    # wrap-safe pole angle error
        thdot_err = x[2] - x_goal[2]              # pole angular velocity error          
        pos_err = x[1] - x_goal[1]                # cart position error
        vel_err   = x[3] - x_goal[3]              # cart velocity error

        # error vector
        e = ca.vertcat(th_err, pos_err, thdot_err, vel_err)

        # total cost
        J = ca.mtimes([e.T, Q, e]) + ca.mtimes([u.T, R, u])

        return J

##############################################################

# example usage
if __name__ == "__main__":

    # define NLP parameters
    nlp_params = NLPSolverParams(
        T       = 7.0,
        dt      = 0.02,
        x_init  = np.array([np.pi*0.01, 0.0, 0.0, 0.0]),
        x_goal  = np.array([np.pi, 0.0, 0.0, 0.0])
    )

    # create cartpole system
    cartpole = Cartpole(dt=nlp_params.dt)

    # create NLP solver
    nlp_solver = NLPSolver(dynamics=cartpole, params=nlp_params)

    # solve the NLP
    nlp_result = nlp_solver.solve()

    # extract results
    X_opt = nlp_result.X_opt
    U_opt = nlp_result.U_opt
    timespan = nlp_result.timespan

    # save results to file
    state_file = "./NLP/results/cartpole_states.csv"
    input_file = "./NLP/results/cartpole_inputs.csv"
    time_file = "./NLP/results/cartpole_time.csv"

    np.savetxt(time_file, timespan)
    np.savetxt(state_file, X_opt)
    np.savetxt(input_file, U_opt)

    print(f"Optimal trajectories saved to:\n{state_file}\n{input_file}\n{time_file}")

    # plot the results
    plt.figure(figsize=(10, 8))
    plt.subplot(5, 1, 1)
    plt.plot(timespan, X_opt[:, 0], label='Pole Angle (rad)', color='orange')
    plt.ylabel('Angle (rad)')
    plt.grid()
    plt.legend()    
    plt.subplot(5, 1, 2)
    plt.plot(timespan, X_opt[:, 1], label='Cart Position (m)')
    plt.ylabel('Position (m)')
    plt.grid()
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(timespan, X_opt[:, 2], label='Pole Angular Velocity (rad/s)', color='red')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.grid()
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(timespan, X_opt[:, 3], label='Cart Velocity (m/s)', color='green')
    plt.ylabel('Velocity (m/s)')
    plt.grid()
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.step(timespan[:-1], U_opt[:], label='Control Input (N)', where='post', color='purple')
    plt.ylabel('Input (N)')
    plt.xlabel('Time (s)')
    plt.grid()
    plt.legend()

    plt.tight_layout()

    # save image of the plot
    plot_file = "./NLP/results/cartpole_trajectories.png"
    plt.savefig(plot_file, dpi=200, bbox_inches="tight")

    plt.show()