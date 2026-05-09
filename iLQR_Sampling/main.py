##
#
# iLQR Sampling for pendulum swing-up
#
##

import os
import torch
import math
import numpy as np
import pendulum
import matplotlib.pyplot as plt
from cost import *
from pendulum import *


##################################################
# iLQR FUNCTIONS
##################################################

def rollout(x0, U, dyn_params, ilqr_params):
    """
    Single-trajectory rollout under the (clamped) control sequence U,
    plus total trajectory cost.

    Args:
        x0:           (nx,) initial state
        U:            (N, nu) control sequence
        dyn_params:   dynamics dict (m, l, b, g, dt, umax)
        ilqr_params:  iLQR dict (unused here, kept for API consistency)

    Returns:
        X:           (N+1, nx) state trajectory
        J:           () scalar tensor, total cost
        U_clamped:   (N, nu) saturated controls actually applied
    """
    # clamp controls (f_disc clamps internally too; expose the clamped values)
    U_clamped = torch.clamp(U, -dyn_params["umax"], dyn_params["umax"])

    # delegate state propagation (B=1 wrap)
    X = pendulum.rollout(x0.unsqueeze(0), U_clamped.unsqueeze(0), dyn_params).squeeze(0)

    # total cost via batched cost_eval (B=1 wrap)
    J = cost_eval(X.unsqueeze(0), U_clamped.unsqueeze(0), dyn_params).squeeze(0)

    return X, J, U_clamped


def linearize_about_trajectory(X, U, dyn_params, ilqr_params):
    """
    Discrete-time Jacobians (Ad_k, Bd_k) of the linearized dynamics at each
    point along a nominal trajectory.

    For each k in 0..N-1:
        Ac_k = df_cont/dx |(X[k],U[k])
        Bc_k = df_cont/du |(X[k],U[k])
    then ZOH-discretize over dt to get Ad_k, Bd_k for the perturbation map
        dx_{k+1} ~ Ad_k dx_k + Bd_k du_k.

    The continuous-time Jacobians are computed either analytically
    (ilqr_params["lin_method"] == "analytical") or via Gaussian-smoothing
    sampling around (X[k], U[k]) (ilqr_params["lin_method"] == "sampling").

    Args:
        X:           (N+1, nx) nominal state trajectory
        U:           (N, nu) nominal control sequence
        dyn_params:  dynamics dict (m, l, b, g, dt)
        ilqr_params: iLQR dict; reads
                       "lin_method": "analytical" (default) or "sampling"
                       "lin_K":      sample count for "sampling"
                       "lin_eps":    perturbation scale for "sampling"

    Returns:
        Ad_list: (N, nx, nx)
        Bd_list: (N, nx, nu)
    """
    # use the first N states (the linearization points for each control step)
    X_lin = X[:-1]                                         # (N, nx)

    method = ilqr_params.get("lin_method", "analytical")

    if method == "analytical":
        # batched analytical Jacobians (treats time axis as the batch axis)
        Ac_list = Ac(X_lin, U, dyn_params)                 # (N, nx, nx)
        Bc_list = Bc(X_lin, U, dyn_params)                 # (N, nx, nu)
        Cc_list = torch.zeros(Ac_list.shape[0], Ac_list.shape[1], 1,
                              dtype=X.dtype, device=X.device)  # iLQR uses only the perturbation map
                                                               # no need for Cc/Cd; pass zeros

    elif method == "sampling":
        # sampling-based linearization (Gaussian smoothing on (xi, eta))
        # linearize_sampling_based reads m,l,b,g from params plus K and eps,
        # so merge the dynamics dict with the sampling knobs at call time
        sampling_params = {**dyn_params,
                           "K":   ilqr_params["lin_K"],
                           "eps": ilqr_params["lin_eps"]}
        Ac_list, Bc_list, _ = linearize_sampling_based(X_lin, U, sampling_params)
        # discard sampled Cc; iLQR only uses the perturbation map
        Cc_list = torch.zeros(Ac_list.shape[0], Ac_list.shape[1], 1,
                              dtype=X.dtype, device=X.device)

    else:
        raise ValueError(
            f"unknown lin_method={method!r}; expected 'analytical' or 'sampling'")

    # ZOH discretization
    Ad_list, Bd_list, _ = discretize_linear_system(Ac_list, Bc_list, Cc_list, dyn_params)

    return Ad_list, Bd_list


def backward_pass(X, U, Ad_list, Bd_list, mu, dyn_params, ilqr_params):
    """
    Riccati-style backward sweep with Levenberg-Marquardt regularization.

    For k = N-1..0, build the local Q-function around (X[k], U[k]) using the
    nominal cost derivatives and the linearized perturbation dynamics
        dx_{k+1} = Ad_k dx_k + Bd_k du_k,
    then compute feedforward / feedback gains
        du_k = k_ff[k] + K_fb[k] dx_k.

    Args:
        X:          (N+1, nx) nominal state trajectory
        U:          (N, nu)   nominal control sequence
        Ad_list:    (N, nx, nx)
        Bd_list:    (N, nx, nu)
        mu:         scalar (LM regularization on Quu)
        dyn_params: dynamics dict (unused; kept for API consistency)
        ilqr_params: iLQR dict (unused; kept for API consistency)

    Returns:
        k_ff_seq: (N, nu)        feedforward gains
        K_fb_seq: (N, nu, nx)    feedback gains
        success:  bool           False if any Quu_reg fails Cholesky (caller bumps mu)
    """
    N      = U.shape[0]
    nx     = X.shape[1]
    nu     = U.shape[1]
    dtype  = X.dtype
    device = X.device

    # bulk-evaluate cost derivatives along the nominal trajectory
    X_stage = X[:-1]              # (N, nx)
    lx_all  = l_x(X_stage)        # (N, nx)
    lxx_all = l_xx(X_stage)       # (N, nx, nx)
    lu_all  = l_u(U)              # (N, nu)
    luu_all = l_uu(U)             # (N, nu, nu)
    lux_all = l_ux(X_stage, U)    # (N, nu, nx)

    # storage for gains
    k_ff_seq = torch.zeros(N, nu,     dtype=dtype, device=device)
    K_fb_seq = torch.zeros(N, nu, nx, dtype=dtype, device=device)

    # At the end of the trajectory, the future cost is just the terminal cost.
    Vx  = lf_x (X[-1:]).squeeze(0)                    # (nx,)
    Vxx = lf_xx(X[-1:]).squeeze(0)                    # (nx, nx)

    # Levenberg-Marquardt regularization matrix, to do Quu_reg = Quu + mu * I
    mu_I = mu * torch.eye(nu, dtype=dtype, device=device)

    # backward sweep
    for k in range(N - 1, -1, -1):
        Ad, Bd = Ad_list[k], Bd_list[k]
        lx,  lu  = lx_all[k],  lu_all[k]
        lxx, luu = lxx_all[k], luu_all[k]
        lux      = lux_all[k]

        # Q-function derivatives:  Q(dx,du) = l(x+dx,u+du) + V(f(x+dx,u+du))
        Qx  = lx  + Ad.T @ Vx
        Qu  = lu  + Bd.T @ Vx
        Qxx = lxx + Ad.T @ Vxx @ Ad # + Vx * f_xx (DDP)
        Qux = lux + Bd.T @ Vxx @ Ad # + Vx * f_ux (DDP)
        Quu = luu + Bd.T @ Vxx @ Bd # + Vx * f_uu (DDP)

        # regularize Quu and check positive-definiteness via Cholesky
        Quu_reg = Quu + mu_I
        try:
            L = torch.linalg.cholesky(Quu_reg)
        except RuntimeError:
            return k_ff_seq, K_fb_seq, False

        # gains:  k_ff = -Quu_reg^{-1} Qu,   K_fb = -Quu_reg^{-1} Qux
        # solve via cholesky solves for numerical stability and speed (Quu_reg is PD by construction)
        k_ff = -torch.cholesky_solve(Qu.unsqueeze(-1), L).squeeze(-1)   # (nu,)
        K_fb = -torch.cholesky_solve(Qux,              L)               # (nu, nx)
        k_ff_seq[k] = k_ff
        K_fb_seq[k] = K_fb

        # value function update (general form, doesn't assume optimal gains)
        Vx  = Qx  + K_fb.T @ Quu_reg @ k_ff + K_fb.T @ Qu  + Qux.T @ k_ff
        Vxx = Qxx + K_fb.T @ Quu_reg @ K_fb + K_fb.T @ Qux + Qux.T @ K_fb
        Vxx = 0.5 * (Vxx + Vxx.T)                    # symmetrize

    return k_ff_seq, K_fb_seq, True


def forward_pass(x0, X_nom, U_nom, k_ff, K_fb, alpha, dyn_params, ilqr_params):
    """
    Closed-loop rollout under the iLQR control law for a single line-search step alpha:
        dx[k]      = x_new[k] - x_nom[k]
        du[k]      = alpha * k_ff[k] + K_fb[k] @ dx[k]
        u_new[k]   = clamp(u_nom[k] + du[k], -umax, umax)
        x_new[k+1] = f_disc(x_new[k], u_new[k])

    Args:
        x0:          (nx,) initial state (typically equal to X_nom[0])
        X_nom:       (N+1, nx) nominal state trajectory
        U_nom:       (N, nu)   nominal control sequence
        k_ff:        (N, nu)     feedforward gains  (from backward_pass)
        K_fb:        (N, nu, nx) feedback gains     (from backward_pass)
        alpha:       scalar line-search step on the feedforward term
        dyn_params:  dynamics dict (m, l, b, g, dt, umax)
        ilqr_params: iLQR dict (unused; kept for API consistency)

    Returns:
        X_new: (N+1, nx)
        U_new: (N, nu)         clamped controls actually applied
        J_new: () scalar tensor
    """
    N, nu = U_nom.shape
    nx    = X_nom.shape[1]
    umax  = dyn_params["umax"]

    X_new    = torch.empty(N + 1, nx, dtype=x0.dtype, device=x0.device)
    U_new    = torch.empty(N,     nu, dtype=x0.dtype, device=x0.device)
    X_new[0] = x0

    xk = x0
    for k in range(N):
        # iLQR control law: open-loop step (scaled by alpha) + closed-loop drift correction
        dx     = xk - X_nom[k]                                       # (nx,)
        du     = alpha * k_ff[k] + K_fb[k] @ dx                      # (nu,)
        uk_new = torch.clamp(U_nom[k] + du, -umax, umax)             # (nu,)
        U_new[k] = uk_new

        # propagate via batched f_disc with B=1
        xk = f_disc(xk.unsqueeze(0), uk_new.unsqueeze(0), dyn_params).squeeze(0)
        X_new[k + 1] = xk

    # total cost via batched cost_eval
    J_new = cost_eval(X_new.unsqueeze(0), U_new.unsqueeze(0), dyn_params).squeeze(0)

    return X_new, U_new, J_new


def ilqr_solve(x0, U_init, dyn_params, ilqr_params):
    """
    Run iLQR to convergence (or max_iter) starting from an initial control guess.

    Outer loop:
        linearize_about_trajectory -> backward_pass -> line search via forward_pass
        -> mu schedule (decrease on accept, increase on reject / Cholesky failure).

    Args:
        x0:          (nx,) initial state
        U_init:      (N, nu) initial control sequence
        dyn_params:  dynamics dict (m, l, b, g, dt, umax)
        ilqr_params: iLQR dict (max_iter, tol, mu, mu_min, mu_max, mu_factor, alphas)

    Returns:
        X:      (N+1, nx) final state trajectory
        U:      (N, nu)   final control sequence
        J_hist: list of floats, total cost after each accepted iteration (J_hist[0] = initial cost)
    """
    max_iter  = ilqr_params["max_iter"]
    tol       = ilqr_params["tol"]
    mu        = ilqr_params["mu"]
    mu_min    = ilqr_params["mu_min"]
    mu_max    = ilqr_params["mu_max"]
    mu_factor = ilqr_params["mu_factor"]
    alphas    = ilqr_params["alphas"]

    # initial rollout
    X, J, U = rollout(x0, U_init, dyn_params, ilqr_params)
    J_hist  = [float(J)]
    print(f"[iLQR] iter   0: J={float(J):.4f}")

    # iterate the iLQR loop
    for it in range(1, max_iter + 1):
        # linearize and run backward pass
        Ad_list, Bd_list = linearize_about_trajectory(X, U, dyn_params, ilqr_params)
        k_ff, K_fb, ok   = backward_pass(X, U, Ad_list, Bd_list, mu, dyn_params, ilqr_params)

        # if backward pass failed -> bump mu and retry
        if not ok:
            mu = min(mu * mu_factor, mu_max)
            print(f"[iLQR] iter {it:3d}: backward pass failed, mu -> {mu:.2e}")
            if mu >= mu_max:
                print(f"[iLQR] mu hit mu_max={mu_max:.2e}; stopping.")
                break
            continue

        # line search: accept the first alpha that improves cost
        accepted = False
        alpha_used = None
        for a in alphas:
            X_try, U_try, J_try = forward_pass(x0, X, U, k_ff, K_fb, a,
                                               dyn_params, ilqr_params)
            if float(J_try) < float(J):
                dJ = float(J) - float(J_try)
                X, U, J    = X_try, U_try, J_try
                accepted   = True
                alpha_used = a
                break
        
        # Good step -> decrease mu, log progress, check convergence
        if accepted:
            mu = max(mu / mu_factor, mu_min)
            J_hist.append(float(J))
            print(f"[iLQR] iter {it:3d}: J={float(J):.4f}  dJ={dJ:.3e}  "
                  f"alpha={alpha_used:.4f}  mu={mu:.2e}")
            if abs(dJ) < tol:
                print(f"[iLQR] converged: |dJ|={abs(dJ):.2e} < tol={tol:.2e}")
                break
        # Bad step -> increase mu and retry (without incrementing iteration counter)
        else:
            mu = min(mu * mu_factor, mu_max)
            print(f"[iLQR] iter {it:3d}: no improvement, mu -> {mu:.2e}")
            if mu >= mu_max:
                print(f"[iLQR] mu hit mu_max={mu_max:.2e}; stopping.")
                break

    print(f"[iLQR] done in {len(J_hist) - 1} accepted iterations. "
          f"J: {J_hist[0]:.4f} -> {J_hist[-1]:.4f}")
    return X, U, J_hist


##################################################
# MAIN
##################################################

if __name__ == "__main__":

    # dynamics parameters
    dyn_params = {
        "m": 1.0,      # mass
        "l": 1.0,      # length
        "b": 0.1,      # damping
        "g": 9.81,     # gravity
        "dt": 0.02,    # time step
        "umax": 3.0,   # max torque
    }

    # iLQR parameters
    ilqr_params = {
        "T": 500,           # horizon (number of control steps)
        "max_iter": 100,    # max outer iterations
        "tol": 1e-4,        # convergence tolerance on cost change
        "mu": 1.0,          # initial Levenberg-Marquardt regularization
        "mu_min": 1e-6,
        "mu_max": 1e10,
        "mu_factor": 2.0,   # multiplicative up/down step on mu
        "alphas": [1.0, 0.75, 0.5, 0.25, 0.125, 0.06, 0.03],  # line search schedule
        # linearization method: "analytical" or "sampling"
        # "lin_method": "analytical",
        "lin_method": "sampling",
        "lin_K":      256,    # samples per linearization (sampling only)
        "lin_eps":    1e-3,   # perturbation scale       (sampling only)
    }

    # initial state (downward at rest)
    x0 = torch.tensor([0.0, 0.0])

    # initial control guess (zero-mean exploration; mirrors MATLAB U = 0.2*randn(1,N))
    torch.manual_seed(0)
    U_init = 0.2 * torch.randn(ilqr_params["T"], 1)

    # run iLQR
    X, U, J_hist = ilqr_solve(x0, U_init, dyn_params, ilqr_params)

    # plotting
    N     = ilqr_params["T"]
    dt    = dyn_params["dt"]
    tspan = torch.arange(N + 1) * dt    # (N+1,)

    fig, axs = plt.subplots(2, 2, figsize=(11, 7))

    # pendulum angle
    axs[0, 0].plot(tspan, X[:, 0], lw=2)
    axs[0, 0].axhline(math.pi, ls="--", c="r", alpha=0.6, label=r"target $\pi$")
    axs[0, 0].set_xlabel("Time (s)"); axs[0, 0].set_ylabel(r"$\theta$ (rad)")
    axs[0, 0].set_title("Pendulum Angle"); axs[0, 0].grid(True); axs[0, 0].legend()

    # pendulum angular velocity
    axs[0, 1].plot(tspan, X[:, 1], lw=2)
    axs[0, 1].axhline(0.0, ls="--", c="r", alpha=0.6, label="target 0")
    axs[0, 1].set_xlabel("Time (s)"); axs[0, 1].set_ylabel(r"$\dot\theta$ (rad/s)")
    axs[0, 1].set_title("Angular Velocity"); axs[0, 1].grid(True); axs[0, 1].legend()

    # control input
    axs[1, 0].step(tspan[:-1], U[:, 0], where="post", lw=2)
    axs[1, 0].axhline( dyn_params["umax"], ls="--", c="k", alpha=0.4)
    axs[1, 0].axhline(-dyn_params["umax"], ls="--", c="k", alpha=0.4)
    axs[1, 0].set_xlabel("Time (s)"); axs[1, 0].set_ylabel("Torque")
    axs[1, 0].set_title("Control Input"); axs[1, 0].grid(True)

    # cost history
    axs[1, 1].plot(range(len(J_hist)), J_hist, "-o", lw=2, ms=4)
    axs[1, 1].set_xlabel("iLQR iteration"); axs[1, 1].set_ylabel("Total cost J")
    axs[1, 1].set_title("Cost History"); axs[1, 1].grid(True)
    axs[1, 1].set_yscale("log")

    fig.tight_layout()

    # save outputs
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    fig.savefig(os.path.join(results_dir, "ilqr_plot.png"), dpi=150)
    np.savetxt(os.path.join(results_dir, "state.csv"), X.numpy(),
               delimiter=",", header="theta,theta_dot", comments="")
    np.savetxt(os.path.join(results_dir, "time.csv"), tspan.numpy(),
               delimiter=",", header="t", comments="")
    print(f"[iLQR] saved plot + state.csv + time.csv -> {results_dir}")

    plt.show()
