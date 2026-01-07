
# standard imports
import numpy as np
import matplotlib.pyplot as plt

# casadi import
import casadi as ca

##############################################################

# pendulum dynamics
def dynamics(x, u):

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

# cost function
def cost(x, u, x_goal):
    w_th    = 10.0
    w_thdot = 2.0
    w_u     = 0.1

    th_err_cost = 1 - ca.cos(x[0] - x_goal[0])   # scalar
    thdot_err   = x[1] - x_goal[1]               # scalar

    J = w_th * th_err_cost + w_thdot * thdot_err**2 + w_u * u[0]**2

    return J

##############################################################

# build the casadi function
x = ca.MX.sym('x', 2) # state: [theta, theta_dot]
u = ca.MX.sym('u', 1) # input: [torque]
xdot = dynamics(x, u)
f_cont = ca.Function('f_cont', [x, u], [xdot])

# ---- sanity check ----
print("f_cont([0,0], [0]) =", f_cont([0, 0], [0]))

dt = 0.01

xk = ca.MX.sym('xk', 2)
uk = ca.MX.sym('uk', 1)

# euler step
xk_next = xk + dt * f_cont(xk, uk)
f_disc = ca.Function('f_disc', [xk, uk], [xk_next])

# ---- sanity check ----
x0 = np.array([0.1, 0.0])   # small angle
u0 = np.array([0.0])        # no torque
print("x1 =", f_disc(x0, u0))

T = 7.0          # time horizon
N = int(T/dt)    # number of steps

# make the optimizer
opti = ca.Opti()

# horizon variable
X = opti.variable(2, N+1)  # state trajectory
U = opti.variable(1, N)    # input trajectory

# initial and goal state constraints
x_init = np.array([0.0, 0.0])  # initial state
x_goal = np.array([np.pi, 0.0]) # goal state
opti.subject_to(X[:,0] == x_init)
opti.subject_to(X[:,N] == x_goal)

# dynamics constraints
for k in range(N):
    opti.subject_to(X[:,k+1] == f_disc(X[:,k], U[:,k]))

# add input constraints
u_max = 2.0
opti.subject_to(opti.bounded(-u_max, U, u_max))

# objective: minimize sum of u^2
J = 0
for k in range(N):
    J += cost(X[:, k], U[:, k], x_goal)
opti.minimize(J)

# initial guess (helps a lot)
opti.set_initial(X, 0)
opti.set_initial(U, 0)

# solve with ipopt (NO fancy options yet)
opti.solver("ipopt")
sol = opti.solve()

X_sol = sol.value(X)
U_sol = sol.value(U)

##############################################################

time = np.arange(0, T, dt)

print(time.shape)
print(X_sol.shape)
print(U_sol.shape)

# # save the results into a CSV file
times_filename = "./NLP/results/pendulum_time.csv"
X_filename = "./NLP/results/pendulum_state.csv"
U_filename = "./NLP/results/pendulum_input.csv"

np.savetxt(times_filename, time)
np.savetxt(X_filename, X_sol.T)
np.savetxt(U_filename, U_sol)

# # plot the results
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(time, X_sol[0, :], label='theta')
# plt.plot(time, X_sol[1, :], label='theta_dot')
# plt.legend()
# plt.subplot(2, 1, 2)
# plt.plot(time[:-1], U_sol[:], label='torque')
# plt.legend()
# plt.show()