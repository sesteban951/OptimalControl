
# standard imports
import numpy as np
import matplotlib.pyplot as plt

# casadi import
import casadi as ca

# custom imports
from nlp import NLPSolver, NLPSolverParams, NLPResult

##############################################################

# Pendulm for NLP example
class Pendulum:

    def __init__(self, dt):

        # state dimension
        self.nx = 2  # [theta, theta_dot]

        # input dimension
        self.nu = 1  # [torque]

        # input limits
        self.u_lb = np.array([-2.0]).reshape(self.nu,1)  # lower input limits
        self.u_ub = np.array([ 2.0]).reshape(self.nu,1)  # upper input limits

        # time step
        self.dt = dt

    # pendulum continuous dynamics
    def f_cont(self, x, u):

        # pendulum parameters
        m = 1.0
        b = 0.1
        g = 9.81
        l = 1.0

        # unpack state
        th =     x[0] # angle
        th_dot = x[1] # angular velocity

        # dynamics equations
        th_ddot = -(g / l) * ca.sin(th) - (b / (m * l**2)) * th_dot + (1 / (m * l**2)) * u

        # dynamics
        xdot = ca.vertcat(th_dot, th_ddot)

        return xdot
    
    # discrete dynamics
    def f_disc(self, xk, uk):

        # euler integration
        xk_next = xk + self.dt * self.f_cont(xk, uk)

        return xk_next
    
    # cost function
    def cost(self, x, u, x_goal):

        # weights
        w_th    = 10.0
        w_thdot = 2.0
        w_u     = 0.1

        # compute error terms
        th_err_cost = 1 - ca.cos(x[0] - x_goal[0])   # scalar
        thdot_err   = x[1] - x_goal[1]               # scalar

        # total cost
        J = w_th * th_err_cost + w_thdot * thdot_err**2 + w_u * u[0]**2

        return J

##############################################################

# example usage
if __name__ == "__main__":

    # define NLP parameters
    nlp_params = NLPSolverParams(
        T       = 7.0,
        dt      = 0.01,
        x_init  = np.array([0.0, 0.0]),
        x_goal  = np.array([np.pi, 0.0])
    )

    # create pendulum system
    pendulum = Pendulum(nlp_params.dt)

    # create NLP solver
    nlp_solver = NLPSolver(dynamics=pendulum, params=nlp_params)

    # solve the NLP
    nlp_result = nlp_solver.solve()

    # extract results
    X_opt = nlp_result.X_opt
    U_opt = nlp_result.U_opt
    timespan = nlp_result.timespan

    # save results to file
    state_file = "./NLP/results/pendulum_states.csv"
    input_file = "./NLP/results/pendulum_inputs.csv"
    time_file = "./NLP/results/pendulum_time.csv"

    np.savetxt(time_file, timespan)
    np.savetxt(state_file, X_opt)
    np.savetxt(input_file, U_opt)

    print(f"Optimal trajectories saved to:\n{state_file}\n{input_file}\n{time_file}")


    # plot the results
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(timespan, X_opt[:, 0], label='theta (rad)', color='orange')
    plt.ylabel('Angle (rad)')
    plt.grid()
    plt.legend()    
    plt.subplot(3, 1, 2)
    plt.plot(timespan, X_opt[:, 1], label='theta_dot (rad/s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.step(timespan[:-1], U_opt[:], label='Control Input (Nm)', where='post', color='purple')
    plt.ylabel('Input (Nm)')
    plt.xlabel('Time (s)')
    plt.grid()
    plt.legend()

    plt.tight_layout()

    # save image of the plot
    plot_file = "./NLP/results/pendulum_trajectories.png"
    plt.savefig(plot_file, dpi=200, bbox_inches="tight")

    plt.show()
