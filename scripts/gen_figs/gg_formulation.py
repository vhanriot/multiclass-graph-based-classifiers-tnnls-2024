import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main.models.graph_models.graph.GG import GG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures" / "gg_formulation"
FIG_DIR.mkdir(parents=True, exist_ok=True)
gg = GG()

np.random.seed(300)
X1 = np.random.multivariate_normal([0.25, 0.25], [[0.01, 0], [0, 0.01]], 5)
X2 = np.random.multivariate_normal([0.5, 0.5], [[0.01, 0], [0, 0.01]], 5)
X = np.vstack((X1, X2))
y = np.hstack((-1 * np.ones(len(X1)), 1 * np.ones(len(X2))))

_, adj = gg.vectorized(X)

pairs = np.unique(np.sort(np.squeeze(np.where(adj == 1)), axis=0), axis=1)

fig, ax = plt.subplots(figsize=(10, 10))
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
_min1, _min2 = np.min(X[:, 0]), np.min(X[:, 1])
_max1, _max2 = np.max(X[:, 0]), np.max(X[:, 1])

_min = np.min([np.min(X[:, 0]), np.min(X[:, 1])]) - 0.1
_max = np.max([np.max(X[:, 0]), np.max(X[:, 1])]) + 0.1

plt.xlim([_min, _max])
plt.ylim([_min, _max])
for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    plt.plot([x1, x2], [y1, y2], c="black", zorder=-1, linewidth=3)

for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    d = math.dist(np.array([x1, y1]), np.array([x2, y2]))
    ax.add_artist(plt.Circle(((x1 + x2) / 2, (y1 + y2) / 2), d / 2, fill=False, ls="--"))

for class_value in np.unique(y):
    row_ix = np.where(y == class_value)
    if class_value == -1:
        plt.scatter(X[row_ix, 0], X[row_ix, 1], facecolors="white", edgecolors="black", s=200)
    else:
        plt.scatter(X[row_ix, 0], X[row_ix, 1], c="black", s=200)

plt.xlabel(r"$\mathbf{x}_{\mathbf{1}}$", fontsize=25)
plt.ylabel(r"$\mathbf{x}_{\mathbf{2}}$", fontsize=25)
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
ax.set_xticks([0.2, 0.6], minor=False)
ax.set_yticks([0.2, 0.6], minor=False)
plt.savefig(FIG_DIR / "gg_formulation.eps", format="eps")
