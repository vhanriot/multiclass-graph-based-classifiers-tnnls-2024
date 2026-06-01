from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main.evaluation import ALL_DATASETS
from main.models.graph_models.graph.GG import GG
from main.models.graph_models.graph.Quality import Quality
from main.models.graph_models.graph.SSV import SSV
from main.preprocessing.normalize_data import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures" / "study_quality"
FIG_DIR.mkdir(parents=True, exist_ok=True)

datasets = [
    "appendicitis",
    "australian",
    "banknote",
    "breastcancer_original",
    "heart",
    "ilpd",
]

sigmas = np.array([0.1, 0.2, 0.3, 0.5, 1, 1e5])
q_cardinality = Quality(quality="cardinality", sigma=None)
gg = GG()
ssv = SSV()
for dataset in datasets:
    dataset_fun = ALL_DATASETS[dataset]
    X, y = dataset_fun(base_path=DATA_DIR)
    X = X.astype(float)
    y = y.astype(int)
    y[y == -1] = 0
    n_classes = len(np.unique(y))
    X, _ = normalize(X_train=X)
    D, adj, _ = gg.vectorized_in(X)
    ssv._get_all_neighbors(adj)
    all_neighbors = ssv.all_neighbors
    qcard = q_cardinality.forward(ssv.all_neighbors, y)
    mean_qcard = np.mean(qcard)
    mean_sigmas = []
    for sigma in sigmas:
        q_dist = Quality(quality="dist_based", sigma=sigma)
        qdist = q_dist.forward(ssv.all_neighbors, y, D)
        mean_sigmas.append(np.mean(qdist))

    fig, ax = plt.subplots(figsize=(20, 2))
    plt.plot(sigmas.astype(str), mean_sigmas, lw=3, marker="o", markersize=8)
    plt.hlines(
        y=[mean_qcard],
        xmin=[f"{sigmas[0]}"],
        xmax=[f"{sigmas[-1]}"],
        linestyles="--",
        lw=3,
    )
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    plt.savefig(FIG_DIR / f"{dataset}.svg")

dataset = "ilpd"
dataset_fun = ALL_DATASETS[dataset]
X, y = dataset_fun(base_path=DATA_DIR)
X = X.astype(float)
y = y.astype(int)
y[y == -1] = 0
n_classes = len(np.unique(y))
X, _ = normalize(X_train=X)
D, adj, _ = gg.vectorized_in(X)
ssv._get_all_neighbors(adj)
all_neighbors = ssv.all_neighbors
qcard = q_cardinality.forward(ssv.all_neighbors, y)
mean_qcard = np.mean(qcard)
mean_sigmas = []
for sigma in sigmas:
    q_dist = Quality(quality="dist_based", sigma=sigma)
    qdist = q_dist.forward(ssv.all_neighbors, y, D)
    mean_sigmas.append(np.mean(qdist))

fig, ax = plt.subplots(figsize=(20, 2))
plt.plot(sigmas.astype(str), mean_sigmas, lw=3, marker="o", markersize=8)
plt.hlines(y=[mean_qcard], xmin=[f"{sigmas[0]}"], xmax=[f"{sigmas[-1]}"], linestyles="--", lw=3)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])

plt.savefig(FIG_DIR / f"{dataset}.svg")
