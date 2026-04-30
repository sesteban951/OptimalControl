import math

import torch
import matplotlib.pyplot as plt

from pendulum import rollout


def main():
    params = {
        "m": 1.0,
        "l": 1.0,
        "b": 0.1,
        "g": 9.81,
        "dt": 0.01,
        "umax": 2.0,
    }

    B = 256          # number of rollouts
    N = 250          # horizon length
    n_plot = 10      # trajectories to plot

    torch.manual_seed(0)

    theta0     = (2.0 * torch.rand(B) - 1.0) * math.pi   # [-pi, pi]
    theta_dot0 = (2.0 * torch.rand(B) - 1.0) * 2.0       # [-2, 2]
    x0 = torch.stack((theta0, theta_dot0), dim=-1)

    U = torch.randn(B, N, 1) * 0.5

    X = rollout(x0, U, params)

    t = torch.arange(N + 1) * params["dt"]
    idx = torch.randperm(B)[:n_plot]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    for i in idx.tolist():
        axes[0].plot(t, X[i, :, 0])
        axes[1].plot(t, X[i, :, 1])

    axes[0].set_ylabel(r"$\theta$ (rad)")
    axes[1].set_ylabel(r"$\dot\theta$ (rad/s)")
    axes[1].set_xlabel("time (s)")
    axes[0].set_title(f"{n_plot} pendulum rollouts (random x0, random u)")
    for ax in axes:
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
