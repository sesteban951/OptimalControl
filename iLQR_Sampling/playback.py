##
#
# Playback animation for the iLQR optimal trajectory.
#
# Loads results/state.csv (theta, theta_dot) and results/time.csv,
# then animates the pendulum on loop.
#
##

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# pendulum visual parameters (must match dyn_params used to generate the trajectory)
L = 1.0   # rod length


def load_trajectory():
    here = os.path.dirname(os.path.abspath(__file__))
    X = np.loadtxt(os.path.join(here, "results", "state.csv"), delimiter=",", skiprows=1)
    t = np.loadtxt(os.path.join(here, "results", "time.csv"),  delimiter=",", skiprows=1)
    return X, t


def main():
    X, t = load_trajectory()
    theta = X[:, 0]
    dt    = float(t[1] - t[0])

    # tip position: theta=0 -> hanging straight down at (0, -L)
    x_tip = L * np.sin(theta)
    y_tip = -L * np.cos(theta)

    # figure
    fig, ax = plt.subplots(figsize=(6, 6))
    pad = 1.25 * L
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # static elements
    ax.plot(0, 0, "ks", ms=10)                 # pivot
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)

    # animated elements
    rod, = ax.plot([], [], "k-", lw=3)
    ball = ax.plot([], [], "ro", ms=18)[0]
    time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                        ha="left", va="top",
                        fontsize=12, family="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

    def init():
        rod.set_data([], [])
        ball.set_data([], [])
        time_text.set_text("")
        return rod, ball, time_text

    def update(k):
        rod.set_data([0.0, x_tip[k]], [0.0, y_tip[k]])
        ball.set_data([x_tip[k]], [y_tip[k]])
        time_text.set_text(f"t = {t[k]:.2f} s")
        return rod, ball, time_text

    # interval is in ms; cap to ~16ms (60fps) if dt is very small
    interval_ms = max(int(dt * 1000), 16)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=interval_ms,
        blit=True,
        repeat=True,
    )

    plt.show()


if __name__ == "__main__":
    main()
