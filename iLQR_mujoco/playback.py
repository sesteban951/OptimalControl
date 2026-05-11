##
#
# Playback the iLQR-optimized trajectory in MuJoCo's passive viewer.
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
    results_dir = os.path.join(here, "results")
    # load the XML name recorded by main.py (e.g. "cartpole.xml") and resolve under models/
    with open(os.path.join(results_dir, "model.txt"), "r") as f:
        xml_name = f.read().strip()
    xml_path    = os.path.join(here, "models", xml_name)

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
