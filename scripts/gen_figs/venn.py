import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main.models.graph_models.graph.GG import GG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures" / "venn"
FIG_DIR.mkdir(parents=True, exist_ok=True)
X = np.array([[0, 0.5], [1, 0.5], [0.5, 0.95]])

gg_class = GG()
_, adj = gg_class.vectorized(X)
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

plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])

for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    d = math.dist(np.array([x1, y1]), np.array([x2, y2]))
    ax.add_artist(plt.Circle(((x1 + x2) / 2, (y1 + y2) / 2), d / 2, fill=True, facecolor="gray"))

d = math.dist(np.array([X[0, 0] + X[0, 1]]), np.array([X[1, 0] + X[1, 1]]))
ax.add_artist(
    plt.Circle(
        ((X[0, 0] + X[1, 0]) / 2, (X[0, 1] + X[1, 1]) / 2),
        d / 2,
        fill=True,
        facecolor="white",
        ls="--",
        edgecolor="black",
    )
)

for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    d = math.dist(np.array([x1, y1]), np.array([x2, y2]))
    ax.add_artist(
        plt.Circle(((x1 + x2) / 2, (y1 + y2) / 2), d / 2, fill=False, facecolor="gray", ls="--", edgecolor="black")
    )

for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    plt.plot([x1, x2], [y1, y2], c="black", zorder=-1, linewidth=3)

plt.scatter(X[:, 0], X[:, 1], c="black", s=800)
for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    plt.plot([x1, x2], [y1, y2], c="black", linewidth=3)

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
plt.text(x=X[0][0] - 0.05, y=X[0][1] - 0.12, s=r"$\mathbf{x}_{\mathbf{j}}$", fontdict=dict(color="black", size=50))

plt.text(x=X[1][0] - 0.05, y=X[1][1] - 0.12, s=r"$\mathbf{x}_{\mathbf{k}}$", fontdict=dict(color="black", size=50))

plt.text(x=X[2][0] - 0.05, y=X[2][1] - 0.12, s=r"$\mathbf{x}_{\mathbf{i}}$", fontdict=dict(color="black", size=50))

plt.savefig(FIG_DIR / "venn.eps")


X = np.array([[0, 0.5], [1, 0.5], [0.5, 0.95]])
gg_class = GG()
_, adj = gg_class.vectorized(X)
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

plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])

d = math.dist(np.array([X[0, 0] + X[0, 1]]), np.array([X[1, 0] + X[1, 1]]))
ax.add_artist(
    plt.Circle(
        ((X[0, 0] + X[1, 0]) / 2, (X[0, 1] + X[1, 1]) / 2),
        d / 2,
        fill=True,
        facecolor="white",
        ls="--",
        edgecolor="black",
    )
)

for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    d = math.dist(np.array([x1, y1]), np.array([x2, y2]))
    ax.add_artist(
        plt.Circle(((x1 + x2) / 2, (y1 + y2) / 2), d / 2, fill=False, facecolor="gray", ls="--", edgecolor="black")
    )

c1 = (X[2] + X[0]) / 2
c2 = (X[2] + X[1]) / 2
c3 = (X[0] + X[1]) / 2
X1 = np.random.multivariate_normal([c1[0], c1[1]], [[0.05, 0], [0, 0.05]], 400)
X2 = np.random.multivariate_normal([c2[0], c2[1]], [[0.05, 0], [0, 0.05]], 400)
X_samples = np.vstack((X1, X2))
d1 = np.sqrt(np.sum((X_samples - c1) ** 2, axis=1))
d1 = d1 < np.sqrt(np.sum((X[2] - c1) ** 2))
d2 = np.sqrt(np.sum((X_samples - c2) ** 2, axis=1))
d2 = d2 < np.sqrt(np.sum((X[2] - c2) ** 2))
d3 = np.sqrt(np.sum((X_samples - c3) ** 2, axis=1))
d3 = d3 > np.sqrt(np.sum((X[0] - c3) ** 2))
valid = 1 * ((d1 | d2) * d3)
X_samples = X_samples[np.where(valid == 1)[0]]
X = np.vstack((X, X_samples))
gg_class = GG()
_, adj = gg_class.vectorized(X)
pairs = np.unique(np.sort(np.squeeze(np.where(adj == 1)), axis=0), axis=1)

for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    plt.plot([x1, x2], [y1, y2], c="black", zorder=-1, linewidth=3)
plt.scatter(X[:, 0], X[:, 1], c="black", s=200)
for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    plt.plot([x1, x2], [y1, y2], c="black", linewidth=3)

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
plt.text(x=X[0][0] - 0.05, y=X[0][1] - 0.12, s=r"$\mathbf{x}_{\mathbf{j}}$", fontdict=dict(color="black", size=50))
plt.text(x=X[1][0] - 0.05, y=X[1][1] - 0.12, s=r"$\mathbf{x}_{\mathbf{k}}$", fontdict=dict(color="black", size=50))
plt.text(x=X[2][0] - 0.05, y=X[2][1] - 0.12, s=r"$\mathbf{x}_{\mathbf{i}}$", fontdict=dict(color="black", size=50))
plt.savefig(FIG_DIR / "venn_gg.eps")
