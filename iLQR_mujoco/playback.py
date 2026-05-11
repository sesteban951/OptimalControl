##
#
# Playback the iLQR-optimized trajectory in MuJoCo's passive viewer.
#
# Loads results/state.csv (cart_pos, pole_angle, cart_vel, pole_vel) and
# results/time.csv, then drives data.qpos / data.qvel through each row and
# calls viewer.sync() to refresh the 3D scene. Loops while the viewer window
# is open. Space toggles pause; close the window to exit.
#
##

import os
import time
import numpy as np
import mujoco
import mujoco.viewer


def load_trajectory(results_dir):
    X = np.loadtxt(os.path.join(results_dir, "state.csv"), delimiter=",", skiprows=1)
    t = np.loadtxt(os.path.join(results_dir, "time.csv"),  delimiter=",", skiprows=1)
    return X, t


def main():
    here        = os.path.dirname(os.path.abspath(__file__))
    xml_path    = os.path.join(here, "models", "cartpole.xml")
    results_dir = os.path.join(here, "results")

    X, t = load_trajectory(results_dir)
    dt   = float(t[1] - t[0])
    N    = len(X)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    nq = model.nq
    assert X.shape[1] == 2 * nq, (
        f"expected state width {2*nq} (=[qpos; qvel]); got {X.shape[1]}"
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for k in range(N):
                if not viewer.is_running():
                    break
                t0 = time.perf_counter()

                # set state from CSV row and refresh derived quantities
                data.qpos[:] = X[k, :nq]
                data.qvel[:] = X[k, nq:]
                mujoco.mj_forward(model, data)

                viewer.sync()

                # sleep to keep playback at real-time (relative to dt)
                elapsed = time.perf_counter() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
