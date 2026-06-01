import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main.models.graph_models.graph.GG import GG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures" / "is_gg"
FIG_DIR.mkdir(parents=True, exist_ok=True)
gg = GG()


X = np.array([[0.8, 0.5], [0.2, 0.5], [0.1, 0.4]])

y = np.array([1, 1, 1])

_, adj = gg.vectorized(X)
pairs = np.unique(np.sort(np.squeeze(np.where(adj == 1)), axis=0), axis=1)

fig, ax = plt.subplots(figsize=(10, 10))
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
top_side = ax.spines["left"]
top_side = ax.spines["bottom"]
_min1, _min2 = np.min(X[:, 0]) - 0.1, np.min(X[:, 1]) - 0.05
_max1, _max2 = np.max(X[:, 0]) + 0.1, np.max(X[:, 1]) + 0.05

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])


_min = np.min([np.min(X[:, 0]), np.min(X[:, 1])]) - 0.1
_max = np.max([np.max(X[:, 0]), np.max(X[:, 1])]) + 0.1

plt.xlim([0, 1])
plt.ylim([0, 1])
for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    plt.plot([x1, x2], [y1, y2], c="black", zorder=-1, linewidth=3)

plt.scatter(X[:, 0], X[:, 1], c="black", s=200)

plt.text(x=X[0][0], y=X[0][1] + 0.02, s=r"$\mathbf{X}_{\mathbf{j}}$", fontdict=dict(color="black", size=20))

plt.text(x=X[1][0], y=X[1][1] + 0.02, s=r"$\mathbf{X}_{\mathbf{k}}$", fontdict=dict(color="black", size=20))


for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    d = math.dist(np.array([x1, y1]), np.array([x2, y2]))
    ax.add_artist(plt.Circle(((x1 + x2) / 2, (y1 + y2) / 2), d / 2, fill=False, ls="--"))

X = np.array([[0.5, 0.5], [0.4, 0.4], [0.3, 0.3]])
y = np.array([1, 1, 1])

_, adj = gg.vectorized(X)
pairs = np.unique(np.sort(np.squeeze(np.where(adj == 1)), axis=0), axis=1)

fig, ax = plt.subplots(figsize=(10, 10))
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
top_side = ax.spines["left"]
top_side.set_visible(False)
top_side = ax.spines["bottom"]
top_side.set_visible(False)
_min1, _min2 = np.min(X[:, 0]) - 0.1, np.min(X[:, 1]) - 0.05
_max1, _max2 = np.max(X[:, 0]) + 0.1, np.max(X[:, 1]) + 0.05

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])


_min = np.min([np.min(X[:, 0]), np.min(X[:, 1])]) - 0.1
_max = np.max([np.max(X[:, 0]), np.max(X[:, 1])]) + 0.1

plt.xlim([0.27, 0.53])
plt.ylim([0.27, 0.53])

x1, x2 = X[0][0], X[1][0]
y1, y2 = X[0][1], X[1][1]
for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    plt.plot([x1, x2], [y1, y2], c="black", zorder=-1, linewidth=5)

plt.scatter(X[:, 0], X[:, 1], c="black", s=200)


for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    d = math.dist(np.array([x1, y1]), np.array([x2, y2]))
    ax.add_artist(plt.Circle(((x1 + x2) / 2, (y1 + y2) / 2), d / 2, fill=False, ls="--"))

plt.text(x=X[0][0], y=X[0][1] + 0.005, s=r"$\mathbf{X}_\mathbf{j}$", fontdict=dict(color="black", size=30))

plt.text(x=X[1][0] + 0.007, y=X[1][1] - 0.006, s=r"$\mathbf{X}_\mathbf{k}$", fontdict=dict(color="black", size=30))

plt.text(x=X[2][0] - 0.01, y=X[2][1] - 0.017, s=r"$\mathbf{X}_\mathbf{i}$", fontdict=dict(color="black", size=30))

x1, x2 = X[0][0], X[1][0]
y1, y2 = X[0][1], X[1][1]
d = math.dist(np.array([x1, y1]), np.array([x2, y2]))
ax.add_artist(plt.Circle(((x1 + x2) / 2, (y1 + y2) / 2), d / 2, fill=False, ls="--"))

plt.savefig(FIG_DIR / "gg_xk_xj_edges.eps", format="eps", bbox_inches="tight")

X = np.array([[0.5, 0.5], [0.4, 0.4], [0.45, 0.45]])
y = np.array([1, 1, 1])

_, adj = gg.vectorized(X)
pairs = np.unique(np.sort(np.squeeze(np.where(adj == 1)), axis=0), axis=1)

fig, ax = plt.subplots(figsize=(10, 10))
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
top_side = ax.spines["left"]
top_side.set_visible(False)
top_side = ax.spines["bottom"]
top_side.set_visible(False)
_min1, _min2 = np.min(X[:, 0]) - 0.1, np.min(X[:, 1]) - 0.1
_max1, _max2 = np.max(X[:, 0]) + 0.1, np.max(X[:, 1]) + 0.1

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])


_min = np.min([np.min(X[:, 0]), np.min(X[:, 1])]) - 0.1
_max = np.max([np.max(X[:, 0]), np.max(X[:, 1])]) + 0.1

plt.xlim([0.37, 0.53])
plt.ylim([0.37, 0.53])

x1, x2 = X[0][0], X[1][0]
y1, y2 = X[0][1], X[1][1]

plt.text(x=X[0][0], y=X[0][1] + 0.005, s=r"$\mathbf{X}_\mathbf{j}$", fontdict=dict(color="black", size=30))

plt.text(x=X[1][0] - 0.007, y=X[1][1] - 0.01, s=r"$\mathbf{X}_\mathbf{k}$", fontdict=dict(color="black", size=30))

plt.text(x=X[2][0] + 0.005, y=X[2][1] - 0.01, s=r"$\mathbf{X}_\mathbf{i}$", fontdict=dict(color="black", size=30))


x1, x2 = X[0][0], X[1][0]
y1, y2 = X[0][1], X[1][1]
d = math.dist(np.array([x1, y1]), np.array([x2, y2]))
ax.add_artist(plt.Circle(((x1 + x2) / 2, (y1 + y2) / 2), d / 2, fill=False, ls="--"))


for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    plt.plot([x1, x2], [y1, y2], c="black", zorder=-1, linewidth=5)

plt.scatter(X[:, 0], X[:, 1], c="black", s=200)


for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    d = math.dist(np.array([x1, y1]), np.array([x2, y2]))
    ax.add_artist(plt.Circle(((x1 + x2) / 2, (y1 + y2) / 2), d / 2, fill=False, ls="--"))
plt.savefig(FIG_DIR / "gg_xk_xj_notedges.eps", format="eps", bbox_inches="tight")
