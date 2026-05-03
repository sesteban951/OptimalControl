##
#
# Pendulum cost functions (stage + terminal)
#
##

import math
import torch


#########################################################
# COST WEIGHTS AND TARGET
#########################################################

# quadratic weights (matches iLQR/pendulum_iLQR.m)
Q_diag = torch.tensor([10.0, 10.0, 1.0])   # state cost weights
R_diag = torch.tensor([0.1])               # control cost weights
Q  = torch.diag(Q_diag)                    # diag(10, 10, 1)
R  = torch.diag(R_diag)                    # diag(0.1)
Qf = Q * 10.0                              # diag(100, 100, 10)

# target state (upright at rest)
x_des = torch.tensor([math.pi, 0.0])

#########################################################
# COST FUNCTION
#########################################################

# stage cost l(x, u) = 0.5 (y - y_des)^T Q (y - y_des) + 0.5 u^T R u
# where y(x) = [cos(theta), sin(theta), theta_dot] (angle-wrap-free)
def l(x, u):
    """
    Args:
        x: (B, 2) state [theta, theta_dot]
        u: (B, 1) torque
    Returns:
        l: (B,) per-batch stage cost
    """
    th = x[..., 0]
    w  = x[..., 1]
    y     = torch.stack((torch.cos(th),       torch.sin(th),       w),         dim=-1)
    y_des = torch.stack((torch.cos(x_des[0]), torch.sin(x_des[0]), x_des[1]))   # (3,) -> broadcasts
    e = y - y_des                                  # (B, 3)

    state_cost   = 0.5 * torch.einsum("bi,ij,bj->b", e, Q, e)
    control_cost = 0.5 * torch.einsum("bi,ij,bj->b", u, R, u)
    return state_cost + control_cost

# output Jacobian J = dy/dx, batched
def _output_jacobian(x):
    th = x[..., 0]
    B = x.shape[0]
    J = torch.zeros(B, 3, 2, dtype=x.dtype, device=x.device)
    J[:, 0, 0] = -torch.sin(th)
    J[:, 1, 0] =  torch.cos(th)
    J[:, 2, 1] = 1.0
    return J

# stage cost gradient w.r.t. state:  l_x = J^T Q (y - y_des)
def l_x(x):
    """
    Args:
        x: (B, 2)
    Returns:
        l_x: (B, 2)
    """
    th = x[..., 0]
    w  = x[..., 1]
    y     = torch.stack((torch.cos(th),       torch.sin(th),       w),         dim=-1)
    y_des = torch.stack((torch.cos(x_des[0]), torch.sin(x_des[0]), x_des[1]))
    e = y - y_des                                  # (B, 3)

    J = _output_jacobian(x)                        # (B, 3, 2)
    return torch.einsum("bji,jk,bk->bi", J, Q, e)  # (B, 2)

# stage cost Hessian w.r.t. state (Gauss-Newton, PSD):  l_xx ≈ J^T Q J
def l_xx(x):
    """
    Args:
        x: (B, 2)
    Returns:
        l_xx: (B, 2, 2)
    """
    J = _output_jacobian(x)                        # (B, 3, 2)
    return torch.einsum("bji,jk,bkl->bil", J, Q, J)  # (B, 2, 2)

# stage cost gradient w.r.t. control:  l_u = R u
def l_u(u):
    """
    Args:
        u: (B, 1)
    Returns:
        l_u: (B, 1)
    """
    return torch.einsum("ij,bj->bi", R, u)         # (B, 1)

# stage cost Hessian w.r.t. control:  l_uu = R
def l_uu(u):
    """
    Args:
        u: (B, 1)
    Returns:
        l_uu: (B, 1, 1)
    """
    B_ = u.shape[0]
    return R.expand(B_, *R.shape)                  # (B, 1, 1)

# stage cost cross-Hessian:  l_ux = d^2 l / du dx = 0  (R independent of x)
def l_ux(x, u):
    """
    Args:
        x: (B, 2)
        u: (B, 1)
    Returns:
        l_ux: (B, 1, 2)
    """
    B_ = x.shape[0]
    m  = u.shape[-1]
    n  = x.shape[-1]
    return torch.zeros(B_, m, n, dtype=x.dtype, device=x.device)

# terminal cost lf(x) = 0.5 (y - y_des)^T Qf (y - y_des)
def lf(x):
    """
    Args:
        x: (B, 2)
    Returns:
        lf: (B,)
    """
    th = x[..., 0]
    w  = x[..., 1]
    y     = torch.stack((torch.cos(th),       torch.sin(th),       w),         dim=-1)
    y_des = torch.stack((torch.cos(x_des[0]), torch.sin(x_des[0]), x_des[1]))
    e = y - y_des                                  # (B, 3)

    return 0.5 * torch.einsum("bi,ij,bj->b", e, Qf, e)

# terminal cost gradient:  lf_x = J^T Qf (y - y_des)
def lf_x(x):
    """
    Args:
        x: (B, 2)
    Returns:
        lf_x: (B, 2)
    """
    th = x[..., 0]
    w  = x[..., 1]
    y     = torch.stack((torch.cos(th),       torch.sin(th),       w),         dim=-1)
    y_des = torch.stack((torch.cos(x_des[0]), torch.sin(x_des[0]), x_des[1]))
    e = y - y_des                                  # (B, 3)

    J = _output_jacobian(x)                        # (B, 3, 2)
    return torch.einsum("bji,jk,bk->bi", J, Qf, e) # (B, 2)

# terminal cost Hessian (Gauss-Newton, PSD):  lf_xx ≈ J^T Qf J
def lf_xx(x):
    """
    Args:
        x: (B, 2)
    Returns:
        lf_xx: (B, 2, 2)
    """
    J = _output_jacobian(x)                        # (B, 3, 2)
    return torch.einsum("bji,jk,bkl->bil", J, Qf, J)  # (B, 2, 2)


#########################################################
# TRAJECTORY COST
#########################################################

# Total cost J = sum_{k=0}^{N-2} l(X[:, k], U[:, k]) + lf(X[:, N-1])
def cost_eval(X, U, params):
    """
    Args:
        X:      (B, N, nx) state trajectory
        U:      (B, N-1, nu) control trajectory
        params: dict (currently unused; kept for API parity)

    Returns:
        J: (B,) total cost per rollout
    """
    B_, N_, nx = X.shape
    nu = U.shape[-1]

    # stage costs: flatten (batch, time) -> single batch axis, then sum back over time
    X_stage = X[:, :-1].reshape(B_ * (N_ - 1), nx)
    U_stage = U.reshape(B_ * (N_ - 1), nu)
    stage = l(X_stage, U_stage).reshape(B_, N_ - 1).sum(dim=1)   # (B,)

    # terminal cost on the final state
    terminal = lf(X[:, -1])                                       # (B,)

    return stage + terminal
