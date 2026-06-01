from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main.models.graph_models.graph.GG import GG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures" / "analysis_filter"
FIG_DIR.mkdir(parents=True, exist_ok=True)
gg = GG()

X = np.array([[0.35, 0.4], [0.65, 0.4], [0.5, 0.38], [0, 0], [0.5, 0.7]])
y = np.array([-1, -1, -1, 1, 1])
X_test = np.array([[0.55, 0.45], [0.55, 0.5], [0.55, 0.65]])
X = np.array([[0.55, 0.7], [0, 0], [0.55, 0.4], [0.7, 0.42], [0.4, 0.42]])

y = np.ones(len(X))
y[0] = -1
y[1] = -1

pos_xi = ["low", "medium", "high"]
for jpos, x_test in enumerate(X_test):
    X[1] = x_test
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
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    _min1, _min2 = np.min(X[:, 0]) - 0.1, np.min(X[:, 1]) - 0.05
    _max1, _max2 = np.max(X[:, 0]) + 0.1, np.max(X[:, 1]) + 0.05

    plt.scatter(X[1, 0], X[1, 1], facecolors="white", edgecolors="black", s=2100, marker="s")

    for class_value in np.unique(y):
        row_ix = np.where(y == class_value)
        if class_value == -1:
            plt.scatter(X[row_ix, 0], X[row_ix, 1], facecolors="white", edgecolors="black", s=2000)
        else:
            plt.scatter(X[row_ix, 0], X[row_ix, 1], c="black", s=2000)

    for i in range(pairs.shape[1]):
        x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
        y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
        plt.plot([x1, x2], [y1, y2], c="black", zorder=-1, linewidth=3)

    _min1, _min2 = np.min(X[:, 0]), np.min(X[:, 1])
    _max1, _max2 = np.max(X[:, 0]), np.max(X[:, 1])

    _min = np.min([np.min(X[:, 0]) - 0.01, np.min(X[:, 1])]) - 0.01
    _max = np.max([np.max(X[:, 0]) + 0.01, np.max(X[:, 1])]) + 0.01

    plt.xlim([_min, _max])
    plt.ylim([_min, _max])

    plt.savefig(FIG_DIR / f"filter_analysis{pos_xi[jpos]}.eps", format="eps")
