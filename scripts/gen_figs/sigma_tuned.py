from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main.models.graph_models.graph.Filter import Filter
from main.models.graph_models.graph.GG import GG
from main.models.graph_models.graph.Quality import Quality
from main.models.graph_models.graph.SSV import SSV

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures" / "sigma_tuned"
FIG_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(5)

gg = GG()
ssv = SSV()
sigma = 0.03

X1 = np.random.multivariate_normal([0.2, 0.2], [[0.02, -0.01], [-0.01, 0.02]], 50)
X2 = np.random.multivariate_normal([0.7, 0.7], [[0.02, -0.01], [-0.01, 0.02]], 50)
X = np.vstack((X1, X2))
y = np.hstack((-1 * np.ones(len(X1)), 1 * np.ones(len(X2))))

out1 = np.array([0.35, 0.85])
out2 = np.array([0.75, 0.35])
out3 = np.array([0.48, 0.2])
out4 = np.array([0.1, 0.55])

outs = np.vstack((out1, out2, out3, out4))
X = np.vstack((X, outs))
y = np.hstack((y, (-1, -1, 1, 1)))

D, adj = gg.vectorized(X)
ssv._build_pairs(adj)
pairs = np.transpose(np.array(np.where(adj == 1)))
pairs = pairs.T

filt = Filter(flex_reg=False)
quality_card = Quality("cardinality", None)
quality_dist = Quality("dist_based", sigma)
ssv._get_all_neighbors(adj)

cardinality_values = quality_card.forward(ssv.all_neighbors, y, D)
dist_based_values = quality_dist.forward(ssv.all_neighbors, y, D)

filtered_cardinality = filt.forward(cardinality_values, y, None)
filtered_dist_based = filt.forward(dist_based_values, y, None)
nv_chip = np.where(filtered_cardinality == 0)
nv_db = np.where(filtered_dist_based == 0)


fig, ax = plt.subplots(figsize=(10, 10))
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    plt.plot([x1, x2], [y1, y2], c="black", zorder=-1, linewidth=3)

plt.scatter(X[nv_chip, 0], X[nv_chip, 1], marker="s", s=500, facecolors="white", edgecolors="black", linewidths=3)

for class_value in np.unique(y):
    row_ix = np.where(y == class_value)
    if class_value == -1:
        plt.scatter(X[row_ix, 0], X[row_ix, 1], facecolors="white", edgecolors="black", s=300)
    else:
        plt.scatter(X[row_ix, 0], X[row_ix, 1], c="black", s=300)

plt.xlabel(r"$\mathbf{x}_{\mathbf{1}}$", fontsize=25)
plt.ylabel(r"$\mathbf{x}_{\mathbf{2}}$", fontsize=25)
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
ax.xaxis.set_major_locator(plt.MaxNLocator(2))
ax.yaxis.set_major_locator(plt.MaxNLocator(2))
plt.savefig(FIG_DIR / "filter_out_chipclass.eps", format="eps")

fig, ax = plt.subplots(figsize=(10, 10))
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    plt.plot([x1, x2], [y1, y2], c="black", zorder=-1, linewidth=3)

plt.scatter(X[nv_db, 0], X[nv_db, 1], marker="s", s=500, facecolors="white", edgecolors="black", linewidths=3)

for class_value in np.unique(y):
    row_ix = np.where(y == class_value)
    if class_value == -1:
        plt.scatter(X[row_ix, 0], X[row_ix, 1], facecolors="white", edgecolors="black", s=300)
    else:
        plt.scatter(X[row_ix, 0], X[row_ix, 1], c="black", s=300)

plt.xlabel(r"$\mathbf{x}_{\mathbf{1}}$", fontsize=25)
plt.ylabel(r"$\mathbf{x}_{\mathbf{2}}$", fontsize=25)
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
ax.xaxis.set_major_locator(plt.MaxNLocator(2))
ax.yaxis.set_major_locator(plt.MaxNLocator(2))
plt.savefig(FIG_DIR / f"filter_out_distbased_sig={sigma}.eps", format="eps")
