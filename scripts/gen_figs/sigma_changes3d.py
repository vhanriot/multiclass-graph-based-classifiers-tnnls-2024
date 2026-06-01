from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main.models.graph_models.graph.ActFun import ActFun
from main.models.graph_models.graph.Distance import Distance
from main.models.graph_models.graph.GG import GG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures" / "sigma_changes3d"
FIG_DIR.mkdir(parents=True, exist_ok=True)
dist = Distance(squared=True)
gg = GG()
act_fun = ActFun("gaussian")
a = 10
y = np.array([-1, -1, -1, 1, 1])

X_test = np.array([[0.55, 0.45], [0.55, 0.5], [0.55, 0.65]])

X = np.array([[0.55, 0.7], [0.55, 0.65], [0.55, 0.4], [0.7, 0.42], [0.4, 0.42]])

y = np.ones(len(X))
y[0] = -1
y[1] = -1

pos_xi = ["low", "medium", "high"]

sigma = 1e-3

sigmas = [1e-2, 1e-2, 1e-1, 1, 1e2]

sigmas = np.linspace(1e-2, 1e-1, 3)

sigmas = [0.055 - 0.02, 0.2]
for sigma in sigmas:
    _min = np.min([np.min(X[:, 0]) - 0.01, np.min(X[:, 1])]) - 0.05
    _max = np.max([np.max(X[:, 0]) + 0.01, np.max(X[:, 1])]) + 0.05

    x1_scale = np.arange(_min, _max, 0.01)
    x2_scale = np.arange(_min, _max, 0.01)

    x_grid, y_grid = np.meshgrid(x1_scale, x2_scale)

    x_g, y_g = x_grid.flatten(), y_grid.flatten()
    x_g, y_g = x_g.reshape((len(x_g), 1)), y_g.reshape((len(y_g), 1))

    grid = np.hstack((x_g, y_g))

    dists = dist.forward(grid, X[1, :].reshape(1, -1)).reshape(
        -1,
    )
    kernel = np.exp(-dists / (2 * sigma**2))
    kernel = np.array(kernel).reshape(x_grid.shape)
    kernel += 1

    _, adj, _ = gg.vectorized_in(X)
    pairs = np.unique(np.sort(np.squeeze(np.where(adj == 1)), axis=0), axis=1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        grid[:, 0].reshape(x_grid.shape),
        grid[:, 1].reshape(x_grid.shape),
        kernel,
        color="gray",
        zorder=0,
    )  # , cmap='viridis')

    ax.scatter(
        X[1, 0],
        X[1, 1],
        0,
        facecolors="white",
        edgecolors="black",
        marker="s",
        zorder=1,
        s=800,
    )

    for class_value in np.unique(y):
        row_ix = np.where(y == class_value)
        if class_value == -1:
            ax.scatter(
                X[row_ix, 0],
                X[row_ix, 1],
                0,
                facecolors="white",
                edgecolors="black",
                s=800,
                depthshade=False,
            )
        else:
            ax.scatter(
                X[row_ix, 0],
                X[row_ix, 1],
                0,
                c="black",
                s=800,
                depthshade=False,
            )

    for i in range(pairs.shape[1]):
        x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
        y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
        ax.plot([x1, x2], [y1, y2], [0, 0], c="black", zorder=1, linewidth=3)

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    top_side = ax.spines["left"]
    top_side.set_visible(False)
    top_side = ax.spines["bottom"]
    top_side.set_visible(False)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    ax.set_xlabel(r"$\mathbf{x}_{\mathbf{1}}$", fontsize=25)
    ax.set_ylabel(r"$\mathbf{x}_{\mathbf{2}}$", fontsize=25)

    ax.view_init(elev=20, azim=-150)

    plt.savefig(FIG_DIR / f"sigma_changes{sigma}.eps", format="eps", bbox_inches="tight", pad_inches=0.0)
