##
#
# Cartpole cost functions.
#
##

import numpy as np


#########################################################
# COST WEIGHTS AND TARGET
#########################################################

# quadratic weights on lifted state and control
Q_diag = np.array([1.0, 10.0, 10.0, 0.1, 0.1])    # [cart_pos, cos, sin, cart_vel, pole_vel]
R_diag = np.array([0.0001])                        # control
Q  = np.diag(Q_diag)                            # (5, 5)
R  = np.diag(R_diag)                            # (1, 1)
Qf = Q * 10.0                                   # terminal weight

# target state: cart at origin, pole upright (theta=0), zero velocities
x_des = np.array([0.0, 0.0, 0.0, 0.0])


#########################################################
# OUTPUT MAPPING y(x) AND ITS JACOBIAN
#########################################################

# lifted state y(x) = [cart_pos, cos(theta), sin(theta), cart_vel, pole_vel]
def _lift(x):
    """
    Args:
        x: (B, 4) state
    Returns:
        y: (B, 5) lifted state
    """
    p     = x[..., 0]
    th    = x[..., 1]
    pdot  = x[..., 2]
    thdot = x[..., 3]
    return np.stack((p, np.cos(th), np.sin(th), pdot, thdot), axis=-1)

# lifted target y_des = y(x_des)
y_des = _lift(x_des[None])[0]

# output Jacobian J = dy/dx
def _output_jacobian(x):
    """
    Args:
        x: (B, 4)
    Returns:
        J: (B, 5, 4)
    """
    B  = x.shape[0]
    th = x[..., 1]
    J = np.zeros((B, 5, 4), dtype=x.dtype)
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = -np.sin(th)
    J[:, 2, 1] =  np.cos(th)
    J[:, 3, 2] = 1.0
    J[:, 4, 3] = 1.0
    return J


#########################################################
# COST FUNCTION
#########################################################

# stage cost l(x, u) = 0.5 (y - y_des)^T Q (y - y_des) + 0.5 u^T R u
def l(x, u):
    """
    Args:
        x: (B, 4) state
        u: (B, 1) control
    Returns:
        l: (B,) per-batch stage cost
    """
    e = _lift(x) - y_des                            # (B, 5)
    state_cost   = 0.5 * np.einsum("bi,ij,bj->b", e, Q, e)
    control_cost = 0.5 * np.einsum("bi,ij,bj->b", u, R, u)
    return state_cost + control_cost

# stage cost gradient w.r.t. state:  l_x = J^T Q (y - y_des)
def l_x(x):
    """
    Args:
        x: (B, 4)
    Returns:
        l_x: (B, 4)
    """
    e = _lift(x) - y_des                            # (B, 5)
    J = _output_jacobian(x)                          # (B, 5, 4)
    return np.einsum("bji,jk,bk->bi", J, Q, e)

# stage cost Hessian w.r.t. state (Gauss-Newton, PSD):  l_xx ≈ J^T Q J
def l_xx(x):
    """
    Args:
        x: (B, 4)
    Returns:
        l_xx: (B, 4, 4)
    """
    J = _output_jacobian(x)                          # (B, 5, 4)
    return np.einsum("bji,jk,bkl->bil", J, Q, J)

# stage cost gradient w.r.t. control:  l_u = R u
def l_u(u):
    """
    Args:
        u: (B, 1)
    Returns:
        l_u: (B, 1)
    """
    return np.einsum("ij,bj->bi", R, u)

# stage cost Hessian w.r.t. control:  l_uu = R
def l_uu(u):
    """
    Args:
        u: (B, 1)
    Returns:
        l_uu: (B, 1, 1)
    """
    B = u.shape[0]
    return np.broadcast_to(R, (B, *R.shape)).copy()

# stage cost cross-Hessian:  l_ux = d^2 l / du dx = 0  (R independent of x)
def l_ux(x, u):
    """
    Args:
        x: (B, 4)
        u: (B, 1)
    Returns:
        l_ux: (B, 1, 4)
    """
    B = x.shape[0]
    return np.zeros((B, u.shape[-1], x.shape[-1]), dtype=x.dtype)


# terminal cost lf(x) = 0.5 (y - y_des)^T Qf (y - y_des)
def lf(x):
    """
    Args:
        x: (B, 4)
    Returns:
        lf: (B,)
    """
    e = _lift(x) - y_des
    return 0.5 * np.einsum("bi,ij,bj->b", e, Qf, e)

# terminal cost gradient:  lf_x = J^T Qf (y - y_des)
def lf_x(x):
    """
    Args:
        x: (B, 4)
    Returns:
        lf_x: (B, 4)
    """
    e = _lift(x) - y_des
    J = _output_jacobian(x)
    return np.einsum("bji,jk,bk->bi", J, Qf, e)

# terminal cost Hessian (Gauss-Newton, PSD):  lf_xx ≈ J^T Qf J
def lf_xx(x):
    """
    Args:
        x: (B, 4)
    Returns:
        lf_xx: (B, 4, 4)
    """
    J = _output_jacobian(x)
    return np.einsum("bji,jk,bkl->bil", J, Qf, J)


#########################################################
# TRAJECTORY COST
#########################################################
 
# total trajectory cost J = sum_{k=0}^{N-1} l(X[k], U[k]) + lf(X[N])
def cost_eval(X, U, params=None):
    """
    Args:
        X:      (N+1, 4) state trajectory
        U:      (N,   1) control sequence
        params: dict (unused; kept for API parity with iLQR_Sampling)
    Returns:
        J: scalar
    """
    stage    = l(X[:-1], U).sum()
    terminal = lf(X[-1:]).item()
    return stage + terminal
