import torch
import numpy as np

##############################################################
# DOUBLE INTEGRATOR
##############################################################

class DoubleIntegrator:

    def __init__(self):

        # dimensions
        self.nx = 4
        self.nu = 2

        # linear matrices
        I_2 = torch.eye(2)
        A = torch.zeros(self.nx, self.nx)
        B = torch.zeros(self.nx, self.nu)
        A[0:2, 2:4] = I_2
        B[2:4, 0:2] = I_2
        self.A = A
        self.B = B

    # continuous time dynamics
    def f_cont(self, x, u):

        # dynamics
        xdot = x @ self.A.T + u @ self.B.T

        return xdot
    
    # discrete time dynamics (RK2)
    def f_disc(self, xk, uk, dt):

        k1 = self.f_cont(xk, uk)
        x1 = xk + 0.5 * dt * k1

        k2 = self.f_cont(x1, uk)
        x2 = xk + dt * k2

        xk_next = x2

        return xk_next
    
    # cost function
    def cost(self, X, U, dt):

        # weights to use
        w_x = torch.tensor([10., 10., 2., 2.])
        w_u = torch.tensor([0.1, 0.1])
        w_du = torch.tensor([0.5, 0.5])
        w_x_f = 100.0 * w_x

        # desired upright position
        x_des_vec = torch.tensor([2.0, 1.0, 0.0, 0.0])  # shape (nx, )
        u_des_vec = torch.tensor([0.0, 0.0])            # shape (nu, )

        # compute the errors terms
        px_err = x_des_vec[0] - X[:, :, 0]
        py_err = x_des_vec[1] - X[:, :, 1]
        vx_err = x_des_vec[2] - X[:, :, 2]
        vy_err = x_des_vec[3] - X[:, :, 3]
        ux_err = u_des_vec[0] - U[:, :, 0]     # shape (K, N)
        uy_err = u_des_vec[1] - U[:, :, 1]     # shape (K, N)

        # compute rate of change of inputs
        dux = (U[:, 1:, 0] - U[:, :-1, 0]) / dt
        duy = (U[:, 1:, 1] - U[:, :-1, 1]) / dt

        # compute the running cost
        cost_stage = (
            w_x[0] * (px_err[:, :-1] ** 2) +
            w_x[1] * (py_err[:, :-1] ** 2) +
            w_x[2] * (vx_err[:, :-1] ** 2) +
            w_x[3] * (vy_err[:, :-1] ** 2)
        )  # (K, N)

        # compute terminal cost
        cost_terminal = (
            w_x_f[0] * (px_err[:, -1] ** 2) +
            w_x_f[1] * (py_err[:, -1] ** 2) +
            w_x_f[2] * (vx_err[:, -1] ** 2) +
            w_x_f[3] * (vy_err[:, -1] ** 2)
        ) # (K, )

        # input cost
        cost_input = (
            w_u[0] * (ux_err ** 2) + 
            w_u[1] * (uy_err ** 2)
        ) # (K, N)
        
        cost_du = (
            w_du[0] * (dux ** 2) +
            w_du[1] * (duy ** 2)
        ) # (K, N-1)

        # compute the cost of each trajectory
        J = (cost_stage.sum(dim=1) * dt
           + cost_terminal
           + cost_input.sum(dim=1) * dt
           + cost_du.sum(dim=1) * dt
        ) # (K, )

        return J
        

##############################################################
# CARTPOLE
##############################################################

class Cartpole:

    def __init__(self):

        # dimensions
        self.nx = 4
        self.nu = 1

        # parameters
        self.mc = 1.0
        self.mp = 0.2
        self.g = 9.81
        self.l = 0.5

    # continuous time dynamics
    def f_cont(self, x, u):

        # system parameters
        mc = self.mc
        mp = self.mp
        g =  self.g
        l =  self.l

        # unpack the state
        th = x[:, 0].unsqueeze(1)     # pole angle       # shape (B, 1)
        th_dot = x[:, 2].unsqueeze(1) # pole angular vel # shape (B, 1)
        p_dot = x[:, 3].unsqueeze(1)  # cart vel         # shape (B, 1)

        # useful quantities
        sin_th = torch.sin(th)       # shape (B, 1)
        cos_th = torch.cos(th)       # shape (B, 1)
        denom = mc + mp * sin_th**2  # shape (B, 1)

        # dynamics equations
        th_ddot = (-u * cos_th - mp * l * th_dot**2 * cos_th * sin_th - (mc + mp) * g * sin_th) / (l * denom)
        p_ddot  = (u + mp * sin_th * (l * th_dot**2 + g * cos_th)) / denom

        # dynamics
        xdot = torch.cat([th_dot, p_dot, th_ddot, p_ddot], dim=1)

        return xdot
    
    # discrete time dynamics (RK2)
    def f_disc(self, xk, uk, dt):

        k1 = self.f_cont(xk, uk)
        x1 = xk + 0.5 * dt * k1

        k2 = self.f_cont(x1, uk)
        x2 = xk + dt * k2

        xk_next = x2

        return xk_next
    
    # cost function
    def cost(self, X, U, dt):

        # weights to use
        w_x = torch.tensor([10., 10., 1., 1.])
        w_u = torch.tensor([0.01])
        w_x_f = 10.0 * w_x

        # desired upright position
        x_des_vec = torch.tensor([torch.pi, 0.0, 0.0, 0.0])  # shape (nx, )
        u_des_vec = torch.tensor([0])                        # shape (nu, )

        # compute the errors terms
        th_err = x_des_vec[0] - X[:, :, 0]    # shape (K, N+1)
        th_err = torch.cos(th_err) - 1        # shape (K, N+1)
        thdot_err = x_des_vec[2] - X[:, :, 2] # shape (K, N+1)
        pos_err = x_des_vec[1] - X[:, :, 1]   # shape (K, N+1)
        vel_err = x_des_vec[3] - X[:, :, 3]   # shape (K, N+1)
        u_err = u_des_vec[0] - U[:, :, 0]     # shape (K, N)

        # compute the running cost
        cost_stage = (
            w_x[0] * (th_err[:, :-1] ** 2) + 
            w_x[1] * (pos_err[:, :-1] ** 2) + 
            w_x[2] * (thdot_err[:, :-1] ** 2) + 
            w_x[3] * (vel_err[:, :-1] ** 2)
        )  # (K, N)

        # compute terminal cost
        cost_terminal = (
            w_x_f[0] * (th_err[:, -1] ** 2) + 
            w_x_f[1] * (pos_err[:, -1] ** 2) + 
            w_x_f[2] * (thdot_err[:, -1] ** 2) + 
            w_x_f[3] * (vel_err[:, -1] ** 2)
        ) # (K, )

        # input cost
        cost_input = (
            w_u[0] * (u_err ** 2)
        ) # (K, N)

        # compute the cost of each trajectory
        J = (cost_stage.sum(dim=1) * dt
           + cost_terminal
           + cost_input.sum(dim=1) * dt
        ) # (K, )

        return J
    
##############################################################

# example usage
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize the class
    # dyn = Cartpole()
    dyn = DoubleIntegrator()

    # settigns
    B = 8
    
    # time settings
    T = 1.0
    dt = 0.05
    N = int(np.round(T/dt))
    