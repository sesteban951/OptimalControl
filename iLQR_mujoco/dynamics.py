import mujoco
import numpy as np
from dataclasses import dataclass


@dataclass
class MJDynamicsConfig:
    xml_path: str
    sim_dt: float
    u_lb: np.ndarray   
    u_ub: np.ndarray   


class MJDynamics:

    def __init__(self, config):
        # copy config to self
        self.config = config

        # load model and create data
        self.init_model()

    # initalize mujoco model
    def init_model(self):
        self.model = mujoco.MjModel.from_xml_path(self.config.xml_path)
        self.data  = mujoco.MjData(self.model)
        print("Model loaded from:", self.config.xml_path)

        # override timestep
        self.model.opt.timestep = self.config.sim_dt

        # dimensions
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu
        self.nx = self.nq + self.nv
        self.dt = float(self.model.opt.timestep)

        # control box (enforced here, not by MuJoCo)
        self.u_lb = np.asarray(self.config.u_lb, dtype=np.float64)
        self.u_ub = np.asarray(self.config.u_ub, dtype=np.float64)
        assert self.u_lb.shape == (self.nu,) and self.u_ub.shape == (self.nu,), (
            f"u_lb/u_ub must have shape ({self.nu},); "
            f"got {self.u_lb.shape} and {self.u_ub.shape}"
        )

    # convienience functions to get current state
    def get_state(self):
        return np.concatenate((self.data.qpos.copy(), self.data.qvel.copy()))

    # convienience function to set state and refresh derived quantities
    def set_state(self, x):
        self.data.qpos[:] = x[:self.nq]
        self.data.qvel[:] = x[self.nq:]
        mujoco.mj_forward(self.model, self.data)

    # discrete step function
    def f_disc(self, x, u, clip=False):
        """
        One mj_step from (x, u).

        Args:
            x:    (nx,) state [qpos; qvel]
            u:    (nu,) control
            clip: if True, saturate u to [u_lb, u_ub] before stepping; if False,
                  pass u through unmodified (MuJoCo no longer clamps either, so
                  this lets callers sample raw, un-clipped dynamics).
        Returns:
            x_next: (nx,) state at t + dt
        """
        # saturate control to config bounds (optional)
        if clip:
            u_eff = np.clip(u, self.u_lb, self.u_ub)
        else:
            u_eff = np.asarray(u, dtype=np.float64)

        # set state and input
        self.set_state(x)
        self.data.ctrl[:] = u_eff

        # step forward
        mujoco.mj_step(self.model, self.data)

        return self.get_state()

    # sampling-based linearization of the discrete map
    def linearize_sampling_based(self, x, u, ilqr_params):
        """
        Estimate (Ad, Bd) for the discrete-time linearization of f_disc at (x, u):
            f_disc(x + dx, u + du) - f_disc(x, u) ≈ Ad dx + Bd du.

        Gaussian smoothing on z = [xi; eta] ~ N(0, I_{nx+nu}):
            y_k = [f_disc(x + eps*xi_k, u + eps*eta_k)
                 - f_disc(x - eps*xi_k, u - eps*eta_k)] / (2*eps)
                ≈ Ad xi_k + Bd eta_k = [Ad | Bd] z_k.
        Monte Carlo least squares:
            [Ad_hat | Bd_hat] = (Y^T Z / K)(Z^T Z / K + reg I)^{-1}.

        Args:
            x:           (nx,) linearization state
            u:           (nu,) linearization control
            ilqr_params: dict; reads
                           "K":   number of paired samples
                           "eps": perturbation scale
                           "reg": ridge on the gram matrix     (optional, default 1e-8)
                           "rng": np.random.Generator          (optional, default fresh)
        Returns:
            Ad: (nx, nx)
            Bd: (nx, nu)
        """
        nx, nu = self.nx, self.nu

        # sampling knobs from ilqr_params
        K   = ilqr_params["K"]
        eps = ilqr_params["eps"]
        reg = ilqr_params.get("reg", 1e-8)
        rng = ilqr_params.get("rng", None)
        if rng is None:
            rng = np.random.default_rng()

        # joint perturbation z = [xi; eta] ~ N(0, I_{nx+nu})
        xi  = rng.standard_normal((K, nx))
        eta = rng.standard_normal((K, nu))

        # central directional derivative y_k ≈ [Ad | Bd] z_k
        Y = np.empty((K, nx), dtype=np.float64)
        for k in range(K):
            f_p = self.f_disc(x + eps * xi[k], u + eps * eta[k], clip=False)
            f_m = self.f_disc(x - eps * xi[k], u - eps * eta[k], clip=False)
            Y[k] = (f_p - f_m) / (2.0 * eps)

        # stacked perturbation Z
        Z = np.concatenate((xi, eta), axis=1)             # (K, nx+nu)

        # least-squares estimate: [Ad | Bd] = (Y^T Z)(Z^T Z + reg I)^{-1}
        YTZ = Y.T @ Z / K                                 # (nx, nx+nu)
        ZTZ = Z.T @ Z / K                                 # (nx+nu, nx+nu)
        G = ZTZ + reg * np.eye(nx + nu)
        AB = np.linalg.solve(G.T, YTZ.T).T                # (nx, nx+nu)

        Ad = AB[:, :nx]
        Bd = AB[:, nx:]
        return Ad, Bd
