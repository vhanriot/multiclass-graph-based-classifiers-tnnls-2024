"""
Baseline models comparison
"""

from pathlib import Path

import catboost as cat
import lightgbm as lgb
import numpy as np
import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from main.evaluation import eval, set_objective_tree_based
from main.evaluation.params import params
from main.models.graph_models.GGRBF import GGRBF
from main.models.ResNet import ResNet
from main.preprocessing.load_data_openml import load
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

CLFS = {
    "ggrbf": {"clf": GGRBF, "params": None},
    "resnet": {"clf": ResNet, "params": params["resnet"]},
    "svm": {"clf": SVC, "params": params["svm"]},
    "random_forest": {"clf": RandomForestClassifier, "params": params["random_forest"]},
    "knn": {"clf": KNeighborsClassifier, "params": params["knn"]},
    "xgboost": {"clf": xgb.XGBClassifier, "params": params["xgboost"]},
    "catboost": {"clf": cat.CatBoostClassifier, "params": params["catboost"]},
    "lightgbm": {"clf": lgb.LGBMClassifier, "params": params["lightgbm"]},
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


def call_eval(dataset, clf_name, tune_with_cv: bool = False, eval_with_cv: bool = False):

    X, y = load(dataset)

    X = X.astype(float)
    y = y.astype(int)
    y[y == -1] = 0
    n_classes = len(np.unique(y))

    # GGRBF
    def ggrbfparam(trial):
        param = {
            "quality": trial.suggest_categorical("quality", ["dist_based"]),
            "sigma": trial.suggest_float("sigma", 0.1, 10, log=True),
            "reg": trial.suggest_categorical("reg", [True]),
            "flex_reg": trial.suggest_categorical("flex_reg", [True]),
            "act_fun": trial.suggest_categorical("act_fun", ["tanhNorm"]),
        }

        for i in range(n_classes):
            param[f"perc{i + 1}"] = (trial.suggest_int(f"perc{i + 1}", 0, 100),)

        return param

    if eval_with_cv:
        skf = split_cv(n_splits=N_FOLDS_EVAL, random_state=SEED, shuffle=True)
        score_vec = []
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f"FOLD NUMBER: {i}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train, X_test = normalize(X_train=X_train, X_test=X_test)

            model = CLFS[clf_name]["clf"]
            print(model)

            if clf_name == "ggrbf":
                params = ggrbfparam
            else:
                params = CLFS[clf_name]["params"]
            func = lambda trial: objective(
                trial,
                X_train,
                y_train,
                n_classes,
                model,
                params,
                tune_with_cv=tune_with_cv,
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

        file_dir = RESULTS_DIR / f"multiclass/{clf_name}"
        Path(file_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{file_dir}/{dataset}.txt", "a+", encoding="utf-8") as f:
            print(rf"{dataset} & {mean_score} $\pm$ {std_score}  ////", file=f)

    else:
        raise ValueError("not implemented.")


if __name__ == "__main__":
    datasets_list = [
        41,
        48,
        61,
        187,
        329,
        679,
        694,
        1100,
        1499,
        1512,
        1523,
        40682,
        41919,
        42261,
        42544,
    ]
    for model in ["knn", "svm", "lightgbm", "random_forest", "xgboost", "resnet", "ggrbf"]:
        for dataset in datasets_list:
            call_eval(dataset, model, tune_with_cv=True, eval_with_cv=True)
