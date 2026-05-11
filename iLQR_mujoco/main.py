##
#
# iLQR with MuJoCo dynamics for cartpole swing-up.
#
##

import os
import numpy as np
import matplotlib.pyplot as plt

from dynamics import MJDynamics, MJDynamicsConfig
from cost import (l_x, l_xx, l_u, l_uu, l_ux,
                  lf_x, lf_xx, cost_eval)


##################################################
# iLQR FUNCTIONS
##################################################

def linearize_about_trajectory(X, U, dyn, ilqr_params):
    """
    Linearize the discrete dynamics at each knot point along the nominal trajectory.

    For k = 0..N-1, estimate (Ad_k, Bd_k) such that the perturbation map satisfies
        dx_{k+1} ≈ Ad_k dx_k + Bd_k du_k.

    Method dispatched by ilqr_params["linearize_method"]:
        "sampling"   -> dyn.linearize_sampling_based (reads "sampling_K", "sampling_eps",
                                                      optional "sampling_reg", "sampling_rng")
        "mujoco_fd"  -> dyn.linearize_mujoco_fd      (reads optional "fd_eps", "fd_centered")

    Args:
        X:           (N+1, nx) nominal state trajectory
        U:           (N, nu)   nominal control sequence
        dyn:         MJDynamics
        ilqr_params: dict
    Returns:
        Ad_list: (N, nx, nx)
        Bd_list: (N, nx, nu)
    """
    N      = U.shape[0]
    nx, nu = dyn.nx, dyn.nu

    method = ilqr_params.get("linearize_method", "sampling")
    if method == "sampling":
        linearize = dyn.linearize_sampling_based
    elif method == "mujoco_fd":
        linearize = dyn.linearize_mujoco_fd
    else:
        raise ValueError(f"unknown linearize_method '{method}'; "
                         f"expected 'sampling' or 'mujoco_fd'")

    Ad_list = np.empty((N, nx, nx))
    Bd_list = np.empty((N, nx, nu))
    for k in range(N):
        Ad, Bd = linearize(X[k], U[k], ilqr_params)
        Ad_list[k] = Ad
        Bd_list[k] = Bd
    return Ad_list, Bd_list


def backward_pass(X, U, Ad_list, Bd_list, mu):
    """
    Riccati-style backward sweep with Levenberg-Marquardt regularization.

    For k = N-1..0, build the local Q-function around (X[k], U[k]) using nominal
    cost derivatives and the perturbation dynamics dx_{k+1} = Ad_k dx_k + Bd_k du_k,
    then compute feedforward / feedback gains
        du_k = k_ff[k] + K_fb[k] dx_k.

    Args:
        X, U:           nominal trajectory and controls
        Ad_list, Bd_list: per-step Jacobians from linearize_about_trajectory
        mu:             scalar LM regularization on Quu

    Returns:
        k_ff_seq: (N, nu)
        K_fb_seq: (N, nu, nx)
        success:  bool  (False if any Quu_reg fails Cholesky → caller bumps mu)
    """
    N      = U.shape[0]
    nx     = X.shape[1]
    nu     = U.shape[1]

    # bulk-evaluate cost derivatives along the nominal trajectory
    X_stage = X[:-1]              # (N, nx)
    lx_all  = l_x (X_stage)       # (N, nx)
    lxx_all = l_xx(X_stage)       # (N, nx, nx)
    lu_all  = l_u (U)             # (N, nu)
    luu_all = l_uu(U)             # (N, nu, nu)
    lux_all = l_ux(X_stage, U)    # (N, nu, nx)

    k_ff_seq = np.zeros((N, nu))
    K_fb_seq = np.zeros((N, nu, nx))

    # terminal cost-to-go
    Vx  = lf_x (X[-1:])[0]                            # (nx,)
    Vxx = lf_xx(X[-1:])[0]                            # (nx, nx)

    mu_I = mu * np.eye(nu)

    for k in range(N - 1, -1, -1):
        Ad, Bd = Ad_list[k], Bd_list[k]
        lx,  lu  = lx_all[k],  lu_all[k]
        lxx, luu = lxx_all[k], luu_all[k]
        lux      = lux_all[k]

        # Q-function derivatives
        Qx  = lx  + Ad.T @ Vx
        Qu  = lu  + Bd.T @ Vx
        Qxx = lxx + Ad.T @ Vxx @ Ad
        Qux = lux + Bd.T @ Vxx @ Ad
        Quu = luu + Bd.T @ Vxx @ Bd

        # regularize Quu and check positive-definiteness via Cholesky
        Quu_reg = Quu + mu_I
        try:
            np.linalg.cholesky(Quu_reg)
        except np.linalg.LinAlgError:
            return k_ff_seq, K_fb_seq, False

        # gains
        k_ff = -np.linalg.solve(Quu_reg, Qu)            # (nu,)
        K_fb = -np.linalg.solve(Quu_reg, Qux)           # (nu, nx)
        k_ff_seq[k] = k_ff
        K_fb_seq[k] = K_fb

        # value-function update (general form, doesn't assume optimal gains)
        Vx  = Qx  + K_fb.T @ Quu_reg @ k_ff + K_fb.T @ Qu  + Qux.T @ k_ff
        Vxx = Qxx + K_fb.T @ Quu_reg @ K_fb + K_fb.T @ Qux + Qux.T @ K_fb
        Vxx = 0.5 * (Vxx + Vxx.T)                       # symmetrize

    return k_ff_seq, K_fb_seq, True


def forward_pass(x0, X_nom, U_nom, k_ff, K_fb, alpha, dyn):
    """
    Closed-loop rollout under the iLQR control law for line-search step alpha:
        dx[k]      = x_new[k] - x_nom[k]
        du[k]      = alpha * k_ff[k] + K_fb[k] @ dx[k]
        u_new[k]   = clamp(u_nom[k] + du[k], u_lb, u_ub)
        x_new[k+1] = f_disc(x_new[k], u_new[k])   # clip=True (physical rollout)

    Returns:
        X_new: (N+1, nx)
        U_new: (N,   nu)   actually-applied (clamped) controls
        J_new: scalar
    """
    N, nu = U_nom.shape
    nx    = X_nom.shape[1]

    X_new    = np.empty((N + 1, nx)); X_new[0] = x0
    U_new    = np.empty((N, nu))
    xk = x0.copy()

    for k in range(N):
        dx = xk - X_nom[k]
        du = alpha * k_ff[k] + K_fb[k] @ dx
        uk = np.clip(U_nom[k] + du, dyn.u_lb, dyn.u_ub)
        U_new[k] = uk
        xk = dyn.f_disc(xk, uk, clip=True)
        X_new[k + 1] = xk

    J_new = cost_eval(X_new, U_new)
    return X_new, U_new, J_new


def ilqr_solve(x0, U_init, dyn, ilqr_params):
    """
    Run iLQR to convergence (or max_iter) starting from an initial control guess.

    Returns:
        X:      (N+1, nx) final state trajectory
        U:      (N, nu)   final control sequence
        J_hist: list of float; J_hist[0] is the initial cost
    """
    N      = U_init.shape[0]
    nx, nu = dyn.nx, dyn.nu

    max_iter  = ilqr_params["max_iter"]
    tol       = ilqr_params["tol"]
    mu        = ilqr_params["mu"]
    mu_min    = ilqr_params["mu_min"]
    mu_max    = ilqr_params["mu_max"]
    mu_factor = ilqr_params["mu_factor"]
    alphas    = ilqr_params["alphas"]

    # initial open-loop rollout of U_init (with clipping)
    X = np.empty((N + 1, nx)); X[0] = x0
    U = np.empty((N, nu))
    xk = x0.copy()
    for k in range(N):
        U[k] = np.clip(U_init[k], dyn.u_lb, dyn.u_ub)
        xk = dyn.f_disc(xk, U[k], clip=True)
        X[k + 1] = xk
    J = cost_eval(X, U)
    J_hist = [float(J)]
    print(f"[iLQR] iter   0: J={float(J):.4f}")

    for it in range(1, max_iter + 1):
        # linearize + backward pass
        Ad_list, Bd_list = linearize_about_trajectory(X, U, dyn, ilqr_params)
        k_ff, K_fb, ok   = backward_pass(X, U, Ad_list, Bd_list, mu)

        # backward pass failed -> bump mu and retry
        if not ok:
            mu = min(mu * mu_factor, mu_max)
            print(f"[iLQR] iter {it:3d}: backward pass failed, mu -> {mu:.2e}")
            if mu >= mu_max:
                print(f"[iLQR] mu hit mu_max={mu_max:.2e}; stopping.")
                break
            continue

        # line search: accept the first alpha that improves cost
        accepted   = False
        alpha_used = None
        dJ         = 0.0
        for a in alphas:
            X_try, U_try, J_try = forward_pass(x0, X, U, k_ff, K_fb, a, dyn)
            if float(J_try) < float(J):
                dJ = float(J) - float(J_try)
                X, U, J    = X_try, U_try, J_try
                accepted   = True
                alpha_used = a
                break

        # good step: decrease mu, log, check convergence
        if accepted:
            mu = max(mu / mu_factor, mu_min)
            J_hist.append(float(J))
            print(f"[iLQR] iter {it:3d}: J={float(J):.4f}  dJ={dJ:.3e}  "
                  f"alpha={alpha_used:.4f}  mu={mu:.2e}")
            if abs(dJ) < tol:
                print(f"[iLQR] converged: |dJ|={abs(dJ):.2e} < tol={tol:.2e}")
                break
        # bad step: increase mu and retry (without incrementing iteration count)
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
    here = os.path.dirname(os.path.abspath(__file__))

    # dynamics
    dyn = MJDynamics(MJDynamicsConfig(
        # xml_path=os.path.join(here, "models", "cartpole.xml"),
        # xml_path=os.path.join(here, "models", "cartpole_walls.xml"),
        xml_path=os.path.join(here, "models", "cartpole_walls_soft.xml"),
        sim_dt=0.01,
        u_lb=np.array([-100.0]),
        u_ub=np.array([ 100.0]),
    ))

    # iLQR parameters
    ilqr_params = {
        "T":         500,
        "max_iter":  250,
        "tol":       1e-6,
        "mu":        1.0,
        "mu_min":    1e-6,
        "mu_max":    1e10,
        "mu_factor": 2.0,
        "alphas":    [1.0, 0.75, 0.5, 0.25, 0.125, 0.06, 0.03],
        # linearization method: "sampling" or "mujoco_fd"
        "linearize_method": "sampling",
        # "linearize_method": "mujoco_fd",
        # sampling-based linearization knobs (consumed by dyn.linearize_sampling_based)
        "sampling_K":   128,
        "sampling_eps": 5e-2,
        "sampling_rng": np.random.default_rng(0),
        # mujoco FD linearization knobs (consumed by dyn.linearize_mujoco_fd)
        "fd_eps":      5e-2,
        "fd_centered": True,
    }

    # initial state: pole-down at rest (qpos[1] = pi in our XML convention)
    x0 = np.array([0.0, np.pi, 0.0, 0.0])

    # initial control guess: sinusoidal energy-pumping pattern (period ~0.5s)
    _t_init = np.arange(ilqr_params["T"]) * dyn.dt
    U_init = 80.0 * np.sin(2.0 * np.pi * 2.0 * _t_init)[:, None] \
             * np.ones((1, dyn.nu))

    # solve
    X, U, J_hist = ilqr_solve(x0, U_init, dyn, ilqr_params)

    # ---- plotting ----
    T_    = ilqr_params["T"]
    tspan = np.arange(T_ + 1) * dyn.dt

    fig, axs = plt.subplots(2, 2, figsize=(11, 7))

    axs[0, 0].plot(tspan, X[:, 0], lw=2)
    axs[0, 0].axhline(0.0, ls="--", c="r", alpha=0.6, label="target 0")
    axs[0, 0].set_xlabel("t (s)"); axs[0, 0].set_ylabel("cart pos (m)")
    axs[0, 0].set_title("Cart position"); axs[0, 0].grid(True); axs[0, 0].legend()

    axs[0, 1].plot(tspan, X[:, 1], lw=2)
    axs[0, 1].axhline(0.0, ls="--", c="r", alpha=0.6, label="upright (theta=0)")
    axs[0, 1].set_xlabel("t (s)"); axs[0, 1].set_ylabel(r"$\theta$ (rad)")
    axs[0, 1].set_title("Pole angle"); axs[0, 1].grid(True); axs[0, 1].legend()

    axs[1, 0].step(tspan[:-1], U[:, 0], where="post", lw=2)
    axs[1, 0].axhline( dyn.u_ub[0], ls="--", c="k", alpha=0.4)
    axs[1, 0].axhline( dyn.u_lb[0], ls="--", c="k", alpha=0.4)
    axs[1, 0].set_xlabel("t (s)"); axs[1, 0].set_ylabel("force (N)")
    axs[1, 0].set_title("Control"); axs[1, 0].grid(True)

    axs[1, 1].plot(range(len(J_hist)), J_hist, "-o", lw=2, ms=4)
    axs[1, 1].set_xlabel("iLQR iteration"); axs[1, 1].set_ylabel("Total cost J")
    axs[1, 1].set_title("Cost history"); axs[1, 1].grid(True)
    axs[1, 1].set_yscale("log")
    fig.tight_layout()

    # ---- save outputs ----
    results_dir = os.path.join(here, "results")
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(os.path.join(results_dir, "ilqr_plot.png"), dpi=150)
    np.savetxt(os.path.join(results_dir, "state.csv"), X,
               delimiter=",", header="cart_pos,pole_angle,cart_vel,pole_vel", comments="")
    np.savetxt(os.path.join(results_dir, "time.csv"), tspan,
               delimiter=",", header="t", comments="")
    # record which XML was used so playback loads the matching model
    with open(os.path.join(results_dir, "model.txt"), "w") as f:
        f.write(os.path.basename(dyn.config.xml_path))
    print(f"[iLQR] saved plot + state.csv + time.csv + model.txt -> {results_dir}")

    plt.show()
