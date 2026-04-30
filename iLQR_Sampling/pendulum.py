##
#
# Pendulum Dynamics (PyTorch, batched)
#
##

import torch


#################################################################
# INVERTED PENDULUM DYNAMICS
#################################################################

def f_cont(x, u, params):
    """
    Continuous-time pendulum dynamics, batched.

    Args:
        x:    (B, 2) tensor   [theta, theta_dot]
        u:    (B, 1) tensor   torque
        params: dict containing system parameters
    Returns:
        returns xdot: (B, 2) tensor
    """
    m = params["m"]  # mass
    b = params["b"]  # damping
    l = params["l"]  # length
    g = params["g"]  # gravity

    u = u.squeeze(-1)

    theta     = x[..., 0]
    theta_dot = x[..., 1]
    theta_ddot = (
        -(g / l) * torch.sin(theta)
        - (b / (m * l**2)) * theta_dot
        + (1.0 / (m * l**2)) * u
    )

    return torch.stack((theta_dot, theta_ddot), dim=-1)


def rk4_step(x, u, params):
    """
    One RK4 step of the continuous dynamics. Fully batched.

    Args:
        x: (B, 2), 
        u: (B, 1), 
    Returns :
        x_next: (B, 2).
    """
    dt = params["dt"]

    u = torch.clamp(u, -params["umax"], params["umax"])

    k1 = f_cont(x,             u, params)
    k2 = f_cont(x + 0.5*dt*k1, u, params)
    k3 = f_cont(x + 0.5*dt*k2, u, params)
    k4 = f_cont(x +     dt*k3, u, params)

    return x + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def rollout(x0, U, params):
    """
    Parallel forward rollouts under the same params.

    ARgs:
        x0: (B, 2)    batch of initial states
        U:  (B, N, 1) batch of control sequences
    Returns:
        X: (B, N+1, 2)
    """
    B, N, _ = U.shape
    X = torch.empty(B, N + 1, 2, dtype=x0.dtype, device=x0.device)
    X[:, 0] = x0
    xk = x0
    for k in range(N):
        xk = rk4_step(xk, U[:, k], params)
        X[:, k + 1] = xk
    return X

