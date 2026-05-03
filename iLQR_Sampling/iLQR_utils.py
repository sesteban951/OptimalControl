##
#
# Sampling ILQR utilities
#
##

import torch
import pendulum


def linearize_sampling_based(x, u, params):
    """
    Estimate (Ac, Bc, Cc) for the affine model
        xdot ≈ Ac x + Bc u + Cc
    at the linearization point (x_bar, u_bar) = (x, u).

    Ac, Bc are estimated jointly via Gaussian smoothing on
        z = [xi; eta] ~ N(0, I_{n+m}):
        y_k = [f(x + eps*xi_k, u + eps*eta_k)
             - f(x - eps*xi_k, u - eps*eta_k)] / (2*eps)
            ≈ Ac xi_k + Bc eta_k = [Ac | Bc] z_k.
    Least-squares Monte Carlo estimate:
        [Ac_hat | Bc_hat] = (sum_k y_k z_k^T)(sum_k z_k z_k^T)^{-1}.

    Cc is then fixed by the affine self-consistency condition:
        Cc = f(x_bar, u_bar) - Ac x_bar - Bc u_bar. (not very accurate since inherits errors from Ac, Bc)

    Args:
        x: (B, n) state
        u: (B, m) control
        params: includes params["eps"] and params["K"] (and optional "reg")

    Returns:
        Ac_hat: (B, n, n)
        Bc_hat: (B, n, m)
        Cc_hat: (B, n, 1)
    """
    K = params["K"]      # number of samples for Gaussian smoothing
    eps = params["eps"]  # perturbation scale
    batch_size, n = x.shape
    m = u.shape[-1]

    # joint perturbation z = [xi; eta] ~ N(0, I_{n+m})
    xi  = torch.randn(batch_size, K, n, dtype=x.dtype, device=x.device)
    eta = torch.randn(batch_size, K, m, dtype=x.dtype, device=x.device)

    # perturb state and control
    x_plus  = x[:, None, :] + eps * xi
    x_minus = x[:, None, :] - eps * xi
    u_plus  = u[:, None, :] + eps * eta
    u_minus = u[:, None, :] - eps * eta

    # flatten batch/sample dimensions for f_cont
    x_plus_flat  = x_plus.reshape(batch_size * K, n)
    x_minus_flat = x_minus.reshape(batch_size * K, n)
    u_plus_flat  = u_plus.reshape(batch_size * K, m)
    u_minus_flat = u_minus.reshape(batch_size * K, m)

    f_plus  = pendulum.f_cont(x_plus_flat,  u_plus_flat,  params).reshape(batch_size, K, n)
    f_minus = pendulum.f_cont(x_minus_flat, u_minus_flat, params).reshape(batch_size, K, n)

    # central directional derivative: y_k ≈ [Ac | Bc] z_k
    directional_derivative = (f_plus - f_minus) / (2.0 * eps)  # (B, K, n)

    # stacked perturbation z = [xi; eta]
    z = torch.cat((xi, eta), dim=-1)  # (B, K, n+m)

    # least-squares estimate: [Ac | Bc] = (sum y z^T)(sum z z^T)^{-1}
    YT_Z = torch.einsum("bki,bkj->bij", directional_derivative, z) / K  # (B, n, n+m)
    ZT_Z = torch.einsum("bki,bkj->bij", z, z) / K                       # (B, n+m, n+m)

    reg = params.get("reg", 1e-8)
    eye = torch.eye(n + m, dtype=x.dtype, device=x.device).expand(batch_size, n + m, n + m)

    G = ZT_Z + reg * eye
    AB_hat = torch.linalg.solve(G.transpose(-1, -2), YT_Z.transpose(-1, -2)).transpose(-1, -2)
    Ac_hat = AB_hat[:, :, :n]
    Bc_hat = AB_hat[:, :, n:]

    # Cc = f(x_bar, u_bar) - Ac x_bar - Bc u_bar  (affine self-consistency)
    f_bar = pendulum.f_cont(x, u, params)                      # (B, n)
    Cc_hat = (f_bar
              - (Ac_hat @ x.unsqueeze(-1)).squeeze(-1)
              - (Bc_hat @ u.unsqueeze(-1)).squeeze(-1)
              ).unsqueeze(-1)                                  # (B, n, 1)

    return Ac_hat, Bc_hat, Cc_hat


#################################################################
# TEST: sampled vs analytical gradients
#################################################################

if __name__ == "__main__":
    import math

    params = {
        "m": 1.0,
        "l": 1.0,
        "b": 0.1,
        "g": 9.81,
        "dt": 0.01,
        "umax": 3.0,
    }

    torch.manual_seed(0)

    B = 64
    theta0     = (2.0 * torch.rand(B) - 1.0) * math.pi
    theta_dot0 = (2.0 * torch.rand(B) - 1.0) * 2.0
    x = torch.stack((theta0, theta_dot0), dim=-1)
    u = (2.0 * torch.rand(B, 1) - 1.0) * params["umax"]

    Ac_true = pendulum.Ac(x, u, params)
    Bc_true = pendulum.Bc(x, u, params)
    Cc_true = pendulum.Cc(x, u, params)

    print(f"{'K':>6}  {'eps':>8}  "
          f"{'Ac rel err':>10}  {'Bc rel err':>10}  {'Cc rel err':>10}")
    print("-" * 60)
    for K in [16, 64, 256, 1024, 4096]:
        for eps in [1e-1, 1e-2, 1e-3, 1e-4]:
            params["K"] = K
            params["eps"] = eps

            Ac_hat, Bc_hat, Cc_hat = linearize_sampling_based(x, u, params)

            def rel(err, true):
                return (err.norm(dim=(-2, -1)) /
                        true.norm(dim=(-2, -1)).clamp_min(1e-12)).mean().item()

            A_rel = rel(Ac_hat - Ac_true, Ac_true)
            B_rel = rel(Bc_hat - Bc_true, Bc_true)
            C_rel = rel(Cc_hat - Cc_true, Cc_true)

            print(f"{K:>6}  {eps:>8.0e}  "
                  f"{A_rel:>10.4e}  {B_rel:>10.4e}  {C_rel:>10.4e}")
