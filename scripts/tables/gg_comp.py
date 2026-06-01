"""Baseline models comparison"""

from pathlib import Path

import numpy as np
import optuna

from main.evaluation import ALL_DATASETS, eval, set_objective_tree_based
from main.models.graph_models.Chipclass import Chipclass
from main.models.graph_models.GGGMM import GGGMM
from main.models.graph_models.GGRBF import GGRBF
from main.preprocessing.normalize_data import normalize
from main.preprocessing.split_data import split_cv, split_nocv
from main.set_seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

N_FOLDS_TUNE = 5
N_FOLDS_EVAL = 5
N_ITER_OPTUNA = 50
SEED = 21

set_seed(SEED)


# GGRBF
def ggrbf(trial):
    param = {
        "quality": trial.suggest_categorical("quality", ["dist_based"]),
        "sigma": trial.suggest_float("sigma", 1e-1, 10, log=True),
        "reg": trial.suggest_categorical("reg", [True]),
        "flex_reg": trial.suggest_categorical("flex_reg", [False]),
        "act_fun": trial.suggest_categorical("act_fun", ["tanhNorm"]),
        "output_weights_algo": trial.suggest_categorical("output_weights_algo", ["pseudoinverse"]),
    }

    return param


def ggrbficann(trial):
    param = {
        "quality": trial.suggest_categorical("quality", ["dist_based"]),
        "sigma": trial.suggest_float("sigma", 1e-1, 10, log=True),
        "reg": trial.suggest_categorical("reg", [True]),
        "flex_reg": trial.suggest_categorical("flex_reg", [False]),
        "act_fun": trial.suggest_categorical("act_fun", ["gaussian"]),
        "output_weights_algo": trial.suggest_categorical("output_weights_algo", ["pseudoinverse"]),
    }

    return param


def gmmgg(trial):
    param = {
        "quality": trial.suggest_categorical("quality", ["dist_based"]),
        "sigma": trial.suggest_float("sigma", 1e-1, 10, log=True),
        "reg": trial.suggest_categorical("reg", [True]),
        "flex_reg": trial.suggest_categorical("flex_reg", [False]),
        "act_fun": trial.suggest_categorical("act_fun", ["gaussian"]),
        "output_weights_algo": trial.suggest_categorical("output_weights_algo", ["pseudoinverse"]),
    }

    return param


def chipclass(trial):
    param = {
        "quality": trial.suggest_categorical("quality", ["dist_based"]),
        "sigma": trial.suggest_float("sigma", 1e-1, 10, log=True),
        "reg": trial.suggest_categorical("reg", [True]),
        "flex_reg": trial.suggest_categorical("flex_reg", [False]),
        "act_fun": trial.suggest_categorical("act_fun", ["tanhNorm"]),
    }

    return param


CLFS = {
    "ggrbf": {"clf": GGRBF, "params": ggrbf},
    "ggrbficann": {"clf": GGRBF, "params": ggrbficann},
    "gmmgg": {"clf": GGGMM, "params": gmmgg},
    "chipclass": {"clf": Chipclass, "params": chipclass},
}


def train_tune(X, y, n_classes, model, params, tune_with_cv):

    model_name = model.__name__

    set_seed()

    set_objective_tree_based(model_name, n_classes, params)

    if tune_with_cv:
        skf = split_cv(n_splits=N_FOLDS_TUNE, random_state=SEED, shuffle=True)
        score_vec = []
        for _i, (train_index, val_index) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            clf = model(**params)
            clf.fit(X_train, y_train)
            score_vec.append(eval(X_val, y_val, n_classes, clf))
        score = np.mean(score_vec)
    else:
        X_train, X_val, y_train, y_val = split_nocv(X, y, test_size=0.3)
        clf = model(**params)
        clf.fit(X_train, y_train)
        score = eval(X_val, y_val, n_classes, clf)
    return score


def objective(trial, X, y, n_classes, model, parameters, tune_with_cv):
    params = parameters(trial)
    score = train_tune(X, y, n_classes, model, params, tune_with_cv)
    return score


def call_eval(dataset: str, clf_name: str, tune_with_cv: bool = False, eval_with_cv: bool = False):

    # -------------- load dataset --------------
    dataset_fun = ALL_DATASETS[dataset]
    X, y = dataset_fun(base_path=DATA_DIR)

    X = X.astype(float)
    y = y.astype(int)
    y[y == -1] = 0
    n_classes = len(np.unique(y))

    if eval_with_cv:
        skf = split_cv(n_splits=N_FOLDS_EVAL, random_state=SEED, shuffle=True)
        score_vec = []
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f"FOLD NUMBER: {i}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train, X_test = normalize(X_train=X_train, X_test=X_test)

            model = CLFS[clf_name]["clf"]
            params = CLFS[clf_name]["params"]
            print(model)

            func = lambda trial: objective(
                trial,
                X_train,
                y_train,
                n_classes,
                model,
                params,
                tune_with_cv,
            )
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=SEED))
            study.optimize(func, n_trials=N_ITER_OPTUNA)
            print("Number of finished trials: ", len(study.trials))
            print("Best trial:", study.best_trial)
            best_params = study.best_params

            set_objective_tree_based(clf_name, n_classes, best_params)

            clf = model(**best_params)
            clf.fit(X_train, y_train)
            score = eval(X_test, y_test, n_classes, clf)

            score_vec.append(score)

        mean_score = np.round(100 * np.mean(score_vec), 4)
        std_score = np.round(100 * np.std(score_vec), 4)

        print(f"SCORE VEC: {score_vec}")

        file_dir = RESULTS_DIR / f"gg_comp/{clf_name}"
        Path(file_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{file_dir}/{dataset}.txt", "a+", encoding="utf-8") as f:
            print(rf"{dataset} & {mean_score} $\pm$ {std_score}  ////", file=f)

    else:
        raise ValueError("not implemented.")


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

    for model in ["chipclass", "gmmgg", "ggrbf", "ggrbficann"]:
        for dataset in datasets_list:
            call_eval(dataset, model, tune_with_cv=True, eval_with_cv=True)
