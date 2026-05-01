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


def f_disc(x, u, params):
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
        xk = f_disc(xk, U[:, k], params)
        X[:, k + 1] = xk
    return X


def Ac(x, u, params):
    """
    Continuous-time Jacobian w.r.t. state, batched.

    Args:
        x:    (B, 2) tensor   [theta, theta_dot]
        u:    (B, 1) tensor   torque
        params: dict containing system parameters
    Returns:
        Ac: (B, 2, 2) tensor
    """
    m = params["m"]  # mass
    b = params["b"]  # damping
    l = params["l"]  # length
    g = params["g"]  # gravity

    theta_bar = x[..., 0]

    Ac_ = torch.zeros(x.shape[0], 2, 2, dtype=x.dtype, device=x.device)
    Ac_[:, 0, 0] = 0.0
    Ac_[:, 0, 1] = 1.0
    Ac_[:, 1, 0] = -(g / l) * torch.cos(theta_bar)
    Ac_[:, 1, 1] = -(b / (m * l**2))

    return Ac_


def Bc(x, u, params):
    """
    Continuous-time Jacobian w.r.t. control, batched.

    Args:
        x:    (B, 2) tensor   [theta, theta_dot]
        u:    (B, 1) tensor   torque
        params: dict containing system parameters
    Returns:
        Bc: (B, 2, 1) tensor
    """
    m = params["m"]  # mass
    l = params["l"]  # length

    Bc_ = torch.zeros(x.shape[0], 2, 1, dtype=x.dtype, device=x.device)
    Bc_[:, 0, 0] = 0.0
    Bc_[:, 1, 0] = (1.0 / (m * l**2))

    return Bc_


def Cc(x, u, params):
    """
    Continuous-time affine offset term, batched. (for original coords., not perturbation coords.)

    Args:
        x:    (B, 2) tensor   [theta, theta_dot], linearization point x_bar
        u:    (B, 1) tensor   torque, linearization point u_bar
        params: dict containing system parameters
    Returns:
        Cc: (B, 2, 1) tensor
    """
    l = params["l"]
    g = params["g"]

    theta_bar = x[..., 0]

    Cc_ = torch.zeros(x.shape[0], 2, 1, dtype=x.dtype, device=x.device)
    Cc_[:, 0, 0] = 0.0
    Cc_[:, 1, 0] = -(g / l) * torch.sin(theta_bar) + (g / l) * theta_bar * torch.cos(theta_bar)

    return Cc_

def discretize_linear_system(Ac_, Bc_, Cc_, params):
    """
    Exact ZOH discretization of continuous-time affine system:
        xdot = Ac x + Bc u + Cc
    into
        x_next = Ad x + Bd u + Cd

    Args:
        Ac_: (B, n, n)
        Bc_: (B, n, m)
        Cc_: (B, n, 1)
        params: dict containing dt

    Returns:
        Ad_: (B, n, n)
        Bd_: (B, n, m)
        Cd_: (B, n, 1)
    """
    dt = params["dt"]

    b, n, _ = Ac_.shape
    m = Bc_.shape[-1]

    M = torch.zeros(b, n + m + 1, n + m + 1, dtype=Ac_.dtype, device=Ac_.device)

    M[:, :n, :n] = Ac_
    M[:, :n, n:n+m] = Bc_
    M[:, :n, n+m:n+m+1] = Cc_

    Md = torch.linalg.matrix_exp(M * dt)

    Ad_ = Md[:, :n, :n]
    Bd_ = Md[:, :n, n:n+m]
    Cd_ = Md[:, :n, n+m:n+m+1]

    return Ad_, Bd_, Cd_