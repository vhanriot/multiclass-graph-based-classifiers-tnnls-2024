from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main.models.graph_models.graph.GG import GG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures" / "plot_ssvs_ses"
FIG_DIR.mkdir(parents=True, exist_ok=True)
seed_exp = 970
np.random.seed(seed_exp)

gg = GG()

X1 = np.random.multivariate_normal([0.3, 0.3], [[0.01, 0], [0, 0.01]], 10)
X2 = np.random.multivariate_normal([0.6, 0.6], [[0.01, 0], [0, 0.01]], 10)
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
X_ssvs = []
mids_ses = []
for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    if y[pairs[0][i]] != y[pairs[1][i]]:
        plt.scatter(X[pairs[0][i], 0], X[pairs[0][i], 1], facecolors="white", edgecolors="black", s=300, marker="s")
        plt.scatter(X[pairs[1][i], 0], X[pairs[1][i], 1], facecolors="white", edgecolors="black", s=300, marker="s")
        plt.plot([x1, x2], [y1, y2], c="black", zorder=-1, linewidth=5)
        X_ssvs.extend((X[pairs[0][i]], X[pairs[1][i]]))
        mids_ses.append(((X[pairs[0][i]] + X[pairs[1][i]]) / 2).tolist())
    else:
        plt.plot([x1, x2], [y1, y2], c="black", zorder=-1, linewidth=1)
for class_value in np.unique(y):
    row_ix = np.where(y == class_value)
    if class_value == -1:
        plt.scatter(X[row_ix, 0], X[row_ix, 1], facecolors="white", edgecolors="black", s=200)
    else:
        plt.scatter(X[row_ix, 0], X[row_ix, 1], c="black", s=200)

X_ssvs = np.unique(X_ssvs, axis=0).tolist()

label_frac_x, label_frac_x_se = 0.6, 0.25
label_frac_y, label_frac_y_se = 0.4, 0.65


chosen_arrow = "-|>"
for i, X_ssv in enumerate(X_ssvs):
    txt_color = "black" if i + 1 == len(X_ssvs) else "white"
    ax.annotate(
        "SSVs",
        xy=(X_ssv[0], X_ssv[1]),
        xycoords="data",
        color=txt_color,
        fontsize=25,
        xytext=(label_frac_x, label_frac_y),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle=chosen_arrow, connectionstyle="arc3,rad=0", fc="w"),
    )

for i, mids_se in enumerate(mids_ses):
    txt_color = "black" if i + 1 == len(mids_ses) else "white"
    ax.annotate(
        "SEs",
        xy=(mids_se[0], mids_se[1]),
        xycoords="data",
        color=txt_color,
        fontsize=25,
        xytext=(label_frac_x_se, label_frac_y_se),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle=chosen_arrow, connectionstyle="arc3,rad=0", fc="w"),
    )

plt.xlabel(r"$\mathbf{x}_{\mathbf{1}}$", fontsize=25)
plt.ylabel(r"$\mathbf{x}_{\mathbf{2}}$", fontsize=25)
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
ax.set_xticks([0.3, 0.7], minor=False)
ax.set_yticks([0.3, 0.7], minor=False)
plt.savefig(FIG_DIR / "gg_ssvs_ses.eps", format="eps")
