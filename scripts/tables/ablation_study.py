"""Chipclass-based models comparison."""

from pathlib import Path

import numpy as np

from main.evaluation import ALL_DATASETS, eval
from main.models.graph_models.Chipclass import Chipclass
from main.preprocessing.normalize_data import normalize
from main.preprocessing.split_data import split_cv
from main.set_seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

sigmas = np.linspace(0.1, 1, 10)
sigmas = np.concatenate((sigmas, [10, 100, 1e6]))

base_params_qd_hk = {
    "quality": "dist_based",
    "reg": True,
    "flex_reg": False,
    "act_fun": "chipclass",
}
base_params_qd_hktanh = {
    "quality": "dist_based",
    "reg": True,
    "flex_reg": False,
    "act_fun": "tanhNorm",
}
params_q_hk = {
    "quality": "cardinality",
    "reg": True,
    "flex_reg": False,
    "act_fun": "chipclass",
    "sigma": None,
}
params_q_hktanh = {
    "quality": "cardinality",
    "reg": True,
    "flex_reg": False,
    "act_fun": "tanhNorm",
    "sigma": None,
}
params_qd_hk = {
    "quality": "dist_based",
    "reg": True,
    "flex_reg": False,
    "act_fun": "chipclass",
}
params_qd_hktanh = {
    "quality": "dist_based",
    "reg": True,
    "flex_reg": False,
    "act_fun": "tanhNorm",
}
col_param_relation = {
    "params_q_hk": params_q_hk,
    "params_q_hktanh": params_q_hktanh,
    "params_qd_hk": params_qd_hk,
    "params_qd_hktanh": params_qd_hktanh,
}
N_FOLDS_TUNE = 5
N_FOLDS_EVAL = 10
N_ITER_OPTUNA = 50
SEED = 21
set_seed(SEED)


def call_eval(dataset: str, column_name: str) -> None:
    """Eval model performance for a given dataset and hyperparameters."""
    # -------------- load dataset --------------
    dataset_fun = ALL_DATASETS[dataset]
    X, y = dataset_fun(base_path=DATA_DIR)
    X = X.astype(float)
    y = y.astype(int)
    y[y == -1] = 0
    n_classes = len(np.unique(y))
    skf = split_cv(n_splits=N_FOLDS_EVAL, random_state=SEED, shuffle=True)
    score_vec = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"FOLD NUMBER: {i}")
        # print(f"FOLD NUMBER: {i}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, X_test = normalize(X_train=X_train, X_test=X_test)
        if column_name in {"params_q_hk", "params_q_hktanh"}:
            params = col_param_relation[column_name]
            clf = Chipclass(**params)
        elif column_name in {"params_qd_hk", "params_qd_hktanh"}:
            params = col_param_relation[column_name]
            best_sigma = None
            best_performance = 0
            for sigma in sigmas:
                if column_name == "params_qd_hk":
                    params = base_params_qd_hk.copy()
                elif column_name == "params_qd_hktanh":
                    params = base_params_qd_hktanh.copy()
                params["sigma"] = sigma
                clf = Chipclass(**params)
                clf.fit(X_train, y_train)
                score = eval(X_test, y_test, n_classes, clf)
                if score > best_performance:
                    best_performance = score
                    best_sigma = sigma
            print(f"BEST SIGMA: {best_sigma}")
            best_params = col_param_relation[column_name]
            best_params["sigma"] = best_sigma
            print(best_params)
            clf = Chipclass(**best_params)
        clf.fit(X_train, y_train)
        score = eval(X_test, y_test, n_classes, clf)
        score_vec.append(score)
    mean_score = np.round(100 * np.mean(score_vec), 4)
    std_score = np.round(100 * np.std(score_vec), 4)
    file_dir = RESULTS_DIR / f"ablation_study/{column_name}"
    Path(file_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{file_dir}/{dataset}.txt", "a+", encoding="utf-8") as f:
        print(rf"{dataset} & {mean_score} $\pm$ {std_score}  ////", file=f)


if __name__ == "__main__":
    datasets_list = [
        "appendicitis",
        "australian",
        "banknote",
        "breastcancer_original",
        "breastcancer_prognostic",
        "climate",
        "fertility",
        "haberman",
        "heart",
        "ilpd",
        "ionosphere",
        "parkinsons",
        "glass_7vsall",
        "vehicle_vanvsall",
        "yeast_5vsall",
        "yeast_9vs1",
        "abalone_18vs9",
    ]
    for dataset in datasets_list:
        call_eval(dataset, "params_q_hk")
        call_eval(dataset, "params_q_hktanh")
        call_eval(dataset, "params_qd_hk")
        call_eval(dataset, "params_qd_hktanh")
