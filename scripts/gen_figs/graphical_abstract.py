from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main.models.graph_models.GGRBF import GGRBF
from main.models.graph_models.graph.Distance import Distance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures" / "graphical_abstract"
FIG_DIR.mkdir(parents=True, exist_ok=True)
params = {"quality": "dist_based", "sigma": 0.1, "reg": True, "flex_reg": True, "act_fun": "tanh"}
clf = GGRBF(**params)
dist_squared = Distance(squared=True)

np.random.seed(1)
X1 = np.random.multivariate_normal([0.25, 0.25], [[0.005, 0], [0, 0.005]], 20)
X2 = np.random.multivariate_normal([0.75, 0.25], [[0.005, 0], [0, 0.005]], 20)
X3 = np.random.multivariate_normal([0.25, 0.75], [[0.005, 0], [0, 0.005]], 20)
X4 = np.random.multivariate_normal([0.75, 0.75], [[0.005, 0], [0, 0.005]], 20)
X = np.vstack((X1, X2, X3, X4))
y = np.hstack((*np.ones(len(X1)), 2 * np.ones(len(X2)), 3 * np.ones(len(X3)), 4 * np.ones(len(X4)))) - 1

out1 = np.array([[0.15, 0.65], [0.34, 0.6], [0.66, 0.1], [0.6, 0.3]])
y1 = np.repeat(0, out1.shape[0])

out2 = np.array([[0.42, 0.1], [0.44, 0.3], [0.63, 0.62], [0.85, 0.6]])
y2 = np.repeat(1, out2.shape[0])

out3 = np.array([[0.55, 0.75], [0.6, 0.65], [0.23, 0.39], [0.41, 0.38]])
y3 = np.repeat(2, out2.shape[0])


out4 = np.array([[0.65, 0.31], [0.9, 0.35], [0.35, 0.9], [0.4, 0.7]])
y4 = np.repeat(3, out3.shape[0])

N_orig = X.shape[0]
X = np.vstack((X, out1, out2, out3, out4))
y = np.hstack((y, y1, y2, y3, y4))

X_all = X.copy()
y_all = y.copy()

y = y.astype(int)
sigma = 0.1

clf._gg(X)
clf.ssv._get_all_neighbors(clf.adj)
quality = clf._quality.forward(clf.ssv.all_neighbors, y, np.sqrt(clf.D))


pairs = np.transpose(np.array(np.where(clf.adj == 1)))
pairs = pairs.T


n_samples_rm = 4
percs = []
for class_value in np.unique(y):
    total_number_of_samples = np.sum(y == class_value)
    percs.append(100 * (n_samples_rm + 1) / total_number_of_samples)

params = {
    "quality": "dist_based",
    "sigma": 0.1,
    "reg": True,
    "flex_reg": True,
    "perc1": percs[0],
    "perc2": percs[1],
    "perc3": percs[2],
    "perc4": percs[3],
    "act_fun": "tanh",
}
clf = GGRBF(**params)
clf._gg(X)
# compute membership function (for filtering procedure)
clf.ssv._get_all_neighbors(clf.adj)
quality = clf._quality.forward(clf.ssv.all_neighbors, y, np.sqrt(clf.D))
# filtering procedure
clf._build_perc_thresholds()
clf.valid_samples = clf.filter.forward(quality, y, clf.percentages)
valid_index = np.where(np.array(clf.valid_samples) == 1)[0]
valid_samples_db = np.array(clf.valid_samples) == 1


nv_db = np.where(valid_samples_db == 0)

fig, ax = plt.subplots(figsize=(10, 10))
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    plt.plot([x1, x2], [y1, y2], c="black", zorder=-1, linewidth=3)

plt.scatter(X[nv_db, 0], X[nv_db, 1], marker="s", s=800, facecolors="white", edgecolors="black", linewidths=3)

dict_class = {0: "black", 1: "white", 2: "black", 3: "white"}
marker_class = {0: "o", 1: "X", 2: "X", 3: "o"}

for class_value in np.unique(y):
    row_ix = np.where(y == class_value)
    plt.scatter(
        X[row_ix, 0],
        X[row_ix, 1],
        facecolors=dict_class[class_value],
        edgecolors="black",
        marker=marker_class[class_value],
        s=500,
    )


plt.xlabel(r"$\mathbf{x}_{\mathbf{1}}$", fontsize=25)
plt.ylabel(r"$\mathbf{x}_{\mathbf{2}}$", fontsize=25)
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)


_min = np.min([np.min(X[:, 0]), np.min(X[:, 1])]) - 0.05
_max = np.max([np.max(X[:, 0]), np.max(X[:, 1])]) + 0.05

plt.xlim([_min, _max])
plt.ylim([_min, _max])
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])

plt.savefig(FIG_DIR / "filtered_samples.svg", dpi=600, transparent=True)

fig, ax = plt.subplots(figsize=(10, 10))
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

for class_value in np.unique(y):
    row_ix = np.where(y == class_value)
    plt.scatter(
        X[row_ix, 0],
        X[row_ix, 1],
        facecolors=dict_class[class_value],
        edgecolors="black",
        marker=marker_class[class_value],
        s=500,
    )

plt.xlabel(r"$\mathbf{x}_{\mathbf{1}}$", fontsize=25)
plt.ylabel(r"$\mathbf{x}_{\mathbf{2}}$", fontsize=25)
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
plt.xlim([_min, _max])
plt.ylim([_min, _max])
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
plt.savefig(FIG_DIR / "samples.svg", dpi=600, transparent=True)

fig, ax = plt.subplots(figsize=(10, 10))
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

for class_value in np.unique(y):
    row_ix = np.where(y == class_value)
    plt.scatter(
        X[row_ix, 0],
        X[row_ix, 1],
        facecolors=dict_class[class_value],
        edgecolors="black",
        marker=marker_class[class_value],
        s=500,
    )

for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    plt.plot([x1, x2], [y1, y2], c="black", zorder=-1, linewidth=3)

plt.xlabel(r"$\mathbf{x}_{\mathbf{1}}$", fontsize=25)
plt.ylabel(r"$\mathbf{x}_{\mathbf{2}}$", fontsize=25)
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
plt.xlim([_min, _max])
plt.ylim([_min, _max])
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
plt.savefig(FIG_DIR / "samplesgg.svg", dpi=600, transparent=True)

_min = np.min([np.min(X[:, 0]) - 0.01, np.min(X[:, 1])]) - 0.05
_max = np.max([np.max(X[:, 0]) + 0.01, np.max(X[:, 1])]) + 0.05

x1_scale = np.arange(_min, _max, 0.01)
x2_scale = np.arange(_min, _max, 0.01)

x_grid, y_grid = np.meshgrid(x1_scale, x2_scale)

x_g, y_g = x_grid.flatten(), y_grid.flatten()
x_g, y_g = x_g.reshape((len(x_g), 1)), y_g.reshape((len(y_g), 1))

grid = np.hstack((x_g, y_g))

kernel_vec = []
for i in range(X.shape[0]):
    dists = dist_squared.forward(grid, X[i].reshape(1, -1)).reshape(
        -1,
    )
    kernel = np.exp(-dists / (2 * sigma**2))
    kernel = np.array(kernel).reshape(x_grid.shape)
    kernel += 1

    kernel_vec.append(kernel)

kernel_acum = np.maximum.reduce(kernel_vec)

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection="3d")

for class_value in np.unique(y):
    row_ix = np.where(y == class_value)
    ax.scatter(
        X[row_ix, 0],
        X[row_ix, 1],
        0,
        facecolors=dict_class[class_value],
        edgecolors="black",
        s=200,
        marker=marker_class[class_value],
        depthshade=False,
    )

for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    ax.plot([x1, x2], [y1, y2], [0, 0], c="black", zorder=1, linewidth=3)

# kernel_acum = kernel_acum/np.max(kernel_acum)+1
ax.plot_surface(grid[:, 0].reshape(x_grid.shape), grid[:, 1].reshape(x_grid.shape), kernel_acum, color="gray", zorder=0)

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

ax.set_xlim([_min, _max])
ax.set_ylim([_min, _max])

ax.set_xlabel(r"$\mathbf{x}_{\mathbf{1}}$", fontsize=25)
ax.set_ylabel(r"$\mathbf{x}_{\mathbf{2}}$", fontsize=25)
ax.view_init(elev=20, azim=-140)

plt.savefig(FIG_DIR / "kernels.svg", dpi=600, bbox_inches="tight", pad_inches=0, transparent=True)

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection="3d")

min_val = -1

for i, x in enumerate(X):
    ax.plot([x[0], x[0]], [x[1], x[1]], [min_val, quality[i]], c="black", zorder=1, linewidth=0.5, linestyle="dashed")

for class_value in np.unique(y):
    row_ix = np.where(y == class_value)
    ax.scatter(
        X[row_ix, 0],
        X[row_ix, 1],
        quality[row_ix],
        facecolors=dict_class[class_value],
        edgecolors="black",
        s=200,
        marker=marker_class[class_value],
        depthshade=False,
    )

for class_value in np.unique(y):
    row_ix = np.where(y == class_value)
    ax.scatter(
        X[row_ix, 0],
        X[row_ix, 1],
        min_val,
        facecolors=dict_class[class_value],
        edgecolors="black",
        s=200,
        marker=marker_class[class_value],
        depthshade=False,
    )

for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    ax.plot([x1, x2], [y1, y2], [min_val, min_val], c="black", zorder=1, linewidth=3)

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

ax.grid(False)
ax.set_xlabel(r"$\mathbf{x}_{\mathbf{1}}$", fontsize=25)
ax.set_ylabel(r"$\mathbf{x}_{\mathbf{2}}$", fontsize=25)
ax.set_zlabel(r"$\mathbf{q}_d$", fontsize=25, rotation=180)
ax.view_init(elev=20, azim=-140)
ax.set_xlim([_min, _max])
ax.set_ylim([_min, _max])
plt.savefig(FIG_DIR / "membershipfun.svg", dpi=600, bbox_inches="tight", pad_inches=0, transparent=True)

X = X[valid_index]
y = y[valid_index]
si = np.where(np.array(clf.valid_samples) == 0)[0]
clf._subgg(si)
clf.ssv.fit(clf.adj, y)
clf.midpoints = (X[clf.ssv.bound_pairs[:, 0]] + X[clf.ssv.bound_pairs[:, 1]]) / 2
Y = y.reshape(-1, 1)
clf.ssv_neg = np.where(Y[clf.ssv.bound_pairs[:, 0]] == 0, X[clf.ssv.bound_pairs[:, 0]], X[clf.ssv.bound_pairs[:, 1]])
clf.ssv_pos = np.where(Y[clf.ssv.bound_pairs[:, 0]] == 0, X[clf.ssv.bound_pairs[:, 1]], X[clf.ssv.bound_pairs[:, 0]])


adj = clf.adj
pairs = np.unique(np.sort(np.squeeze(np.where(adj == 1)), axis=0), axis=1)

fig, ax = plt.subplots(figsize=(10, 10))
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

dict_class = {0: "black", 1: "white", 2: "black", 3: "white"}
marker_class = {0: "o", 1: "X", 2: "X", 3: "o"}

for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    plt.plot([x1, x2], [y1, y2], c="black", zorder=-1, linewidth=3)

for class_value in np.unique(y):
    row_ix = np.where(y == class_value)
    plt.scatter(
        X[row_ix, 0],
        X[row_ix, 1],
        facecolors=dict_class[class_value],
        edgecolors="black",
        marker=marker_class[class_value],
        s=500,
    )

X_ssvs = []
for i in range(pairs.shape[1]):
    x1, x2 = X[pairs[0][i]][0], X[pairs[1][i]][0]
    y1, y2 = X[pairs[0][i]][1], X[pairs[1][i]][1]
    if y[pairs[0][i]] != y[pairs[1][i]]:
        plt.scatter(
            X[pairs[0][i], 0], X[pairs[0][i], 1], facecolors="white", edgecolors="black", s=800, marker="s", zorder=-1
        )
        plt.scatter(
            X[pairs[1][i], 0], X[pairs[1][i], 1], facecolors="white", edgecolors="black", s=800, marker="s", zorder=-1
        )
        X_ssvs.extend((X[pairs[0][i]], X[pairs[1][i]]))
X_ssvs = np.unique(X_ssvs, axis=0).tolist()

for i, x in enumerate(X_ssvs):
    x = np.array(x)
    plt.text(x[0] + 0.02, x[1] + 0.02, rf"$\zeta_{i + 1}$", fontsize=30)

plt.xlabel(r"$\mathbf{x}_{\mathbf{1}}$", fontsize=25)
plt.ylabel(r"$\mathbf{x}_{\mathbf{2}}$", fontsize=25)
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
ax.xaxis.set_major_locator(plt.MaxNLocator(2))
ax.yaxis.set_major_locator(plt.MaxNLocator(2))
_min = np.min([np.min(X[:, 0]), np.min(X[:, 1])]) - 0.05
_max = np.max([np.max(X[:, 0]), np.max(X[:, 1])]) + 0.05

plt.xlim([_min, _max])
plt.ylim([_min, _max])
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
plt.savefig(FIG_DIR / "finalsvs.svg", transparent=True)
