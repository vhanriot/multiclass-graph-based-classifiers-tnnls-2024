import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main.evaluation import ALL_DATASETS
from main.models.graph_models.graph.GG import GG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures" / "study_recgg_time"
FIG_DIR.mkdir(parents=True, exist_ok=True)
datasets = [
    "appendicitis",
    "australian",
    "banknote",
    "breastcancer_original",
    "heart",
    "ilpd",
]


def generate_random_list(total_size, n):
    # Ensure n is not greater than the range of possible unique numbers
    assert n < total_size
    return random.sample(range(total_size), n)


class Timer:
    def __init__(self):
        self.save_times = []
        self.start_time = 0

    def start(self):
        self.start_time = time.process_time()

    def end(self):
        end_time = time.process_time()
        self.save_times.append(end_time - self.start_time)

    def get_average_time(self):
        return np.mean(self.save_times)

    def get_std_time(self):
        return np.std(self.save_times)


orig_timers = {}
our_timers = {}

percentages_maintained_samples = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
times_orig = {}
times_our = {}

N_experiments = 10
for dataset in datasets:
    gg = GG()
    print(f"Dataset: {dataset}")
    dataset_fun = ALL_DATASETS[dataset]
    X, y = dataset_fun(base_path=DATA_DIR)
    X = X.astype(float)

    D, ADJ, ADJ_IN = gg.vectorized_in(X)

    orig_timers[dataset] = {}
    our_timers[dataset] = {}

    times_orig[dataset] = {}
    times_our[dataset] = {}
    for percentage in percentages_maintained_samples:
        print(f"Percentage: {percentage}")

        orig_timers[dataset][percentage] = Timer()
        our_timers[dataset][percentage] = Timer()
        maintained_samples = np.sort(np.array(generate_random_list(X.shape[0], int(X.shape[0] * percentage))))
        removed_samples = np.array([elem for elem in np.arange(X.shape[0]) if elem not in maintained_samples])
        X_filtered = X[maintained_samples].copy()

        for j in range(N_experiments):
            orig_timers[dataset][percentage].start()
            _, adj_orig, _ = gg.vectorized_in(X_filtered)
            orig_timers[dataset][percentage].end()

            our_timers[dataset][percentage].start()
            adj_our = gg.sub_GG(ADJ_IN, D, removed_samples)
            our_timers[dataset][percentage].end()

            if (adj_orig == adj_our).all():
                continue
            else:
                raise ValueError(f"original adjacency is different from ours at dataset {dataset} experiment {j}")

        times_orig[dataset][percentage] = {
            "mean": orig_timers[dataset][percentage].get_average_time(),
            "std": orig_timers[dataset][percentage].get_std_time(),
        }
        times_our[dataset][percentage] = {
            "mean": our_timers[dataset][percentage].get_average_time(),
            "std": our_timers[dataset][percentage].get_std_time(),
        }


percentages_removed_samples = 1 - np.array(percentages_maintained_samples)
percentages_removed_samples = np.round(100 * percentages_removed_samples)
percentages_removed_samples = percentages_removed_samples.astype(int).astype(str)


N_datasets = len(datasets)

for i, dataset in enumerate(datasets):
    fig, ax = plt.subplots(figsize=(4, 10))
    data_dict = times_orig[dataset]

    means = np.array([data_dict[perc]["mean"] for perc in data_dict]).reshape(-1, 1)
    stds = np.array([data_dict[perc]["std"] for perc in data_dict]).reshape(-1, 1)

    color = plt.cm.viridis(i / N_datasets)  # Use a colormap to get different colors for each line

    plt.plot(
        percentages_removed_samples,
        means,
        color="navy",
        label="standard",
        marker="o",
        markersize=10,
    )
    plt.fill_between(
        percentages_removed_samples,
        (means + stds).reshape(-1),
        (means - stds).reshape(-1),
        color="navy",
        alpha=0.3,
    )

    data_dict = times_our[dataset]

    means = np.array([data_dict[perc]["mean"] for perc in data_dict]).reshape(-1, 1)
    stds = np.array([data_dict[perc]["std"] for perc in data_dict]).reshape(-1, 1)

    color = plt.cm.viridis(i / N_datasets)  # Use a colormap to get different colors for each line

    plt.plot(
        percentages_removed_samples,
        means,
        color="orange",
        label="proposed",
        marker="s",
        markersize=10,
    )
    plt.fill_between(
        percentages_removed_samples,
        (means + stds).reshape(-1),
        (means - stds).reshape(-1),
        color="orange",
        alpha=0.3,
        linestyle="--",
    )
    # ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    if dataset == datasets[-1]:
        plt.legend(fontsize="20")
    plt.savefig(FIG_DIR / f"{dataset}.svg")
