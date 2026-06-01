import os
from pathlib import Path

import catboost as cat
import lightgbm as lgb
import numpy as np
import optuna
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from main.hyperparameters_spaces import hyperparameters
from main.models.graph_models.Chipclass import Chipclass
from main.models.graph_models.GGGMM import GGGMM
from main.models.graph_models.GGRBF import GGRBF
from main.models.graph_models.graph.GG import GG
from main.preprocessing.load_data_openml import load
from main.preprocessing.normalize_data import normalize
from main.preprocessing.split_data import split_nocv
from main.set_seed import set_seed

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ':16:8
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
CLFS = {
    "chipclass": {"clf": Chipclass, "params": hyperparameters["chipclass"]},
    "gggmm": {"clf": GGGMM, "params": hyperparameters["gggmm"]},
    "ggrbf": {"clf": GGRBF, "params": hyperparameters["ggrbf"]},
    "svm": {"clf": SVC, "params": hyperparameters["svm"]},
    "random_forest": {"clf": RandomForestClassifier, "params": hyperparameters["random_forest"]},
    "knn": {"clf": KNeighborsClassifier, "params": hyperparameters["knn"]},
    "xgboost": {"clf": xgb.XGBClassifier, "params": hyperparameters["xgboost"]},
    "catboost": {"clf": cat.CatBoostClassifier, "params": hyperparameters["catboost"]},
    "lightgbm": {"clf": lgb.LGBMClassifier, "params": hyperparameters["lightgbm"]},
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_FOLDS_TUNE = 5
N_FOLDS_EVAL = 10
N_ITER_OPTUNA = 50
SEED = 21
set_seed(SEED)


def eval(X, y, n_classes, clf):
    # -------------- compute AUC --------------
    if n_classes == 2:
        yhat = clf.predict_proba(X)[:, 1]
        score = roc_auc_score(y_true=y, y_score=yhat)
    else:
        yhat = clf.predict_proba(X)
        score = roc_auc_score(y_true=y, y_score=yhat, multi_class="ovo", average="macro")
    return score


def compute_adjs(X, y, classifier, file_path, params):

    if classifier not in ["chipclass", "gggmm", "ggrbf"]:
        return

    # -------------- computing gabriel graph --------------
    if not os.path.exists(file_path):
        print("computing GG")
        gabrielgraph = GG()
        F, adj, adj_in = gabrielgraph.vectorized_in(X)
        Path(file_path).mkdir(parents=True, exist_ok=True)
        np.save(f"{file_path}/F.npy", np.array(F))
        np.save(f"{file_path}/adj.npy", np.array(adj))
        np.save(f"{file_path}/adj_in.npy", np.array(adj_in))
    else:
        F = np.load(f"{file_path}/F.npy")
        adj = np.load(f"{file_path}/adj.npy")
        adj_in = np.load(f"{file_path}/adj_in.npy")

    params["D"] = F
    params["adj"] = adj
    params["adj_in"] = adj_in


def set_objective_tree_based(clf_name, n_classes, params):
    # for tree-based models, following the metrics and objective from https://github.com/LeoGrin/tabular-benchmark
    if clf_name == "XGBClassifier":
        if n_classes == 2:
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "auc"
        else:
            params["objective"] = "multi:softprob"
            params["num_class"] = n_classes
            params["eval_metric"] = "mlogloss"
    elif clf_name == "CatBoostClassifier":
        # params["eval_metric"] = 'AUC'
        params["verbose"] = False
        params["allow_writing_files"] = False
        # params['task_type'] = "GPU"
    elif clf_name == "LGBMClassifier":
        params["verbosity"] = -1
        if n_classes == 2:
            params["objective"] = "binary"
            params["metric"] = "auc"
        else:
            params["objective"] = "multiclass"
            params["num_class"] = n_classes
            params["metric"] = "multiclass"


def train_tune(dataset, X, y, n_classes, model, params, CV):

    model_name = model.__name__

    set_seed(SEED)

    set_objective_tree_based(model_name, n_classes, params)

    # -------------- IF CROSS-VALIDATION IS FALSE --------------
    if not CV:
        # -------------- creating training and validation splits --------------
        # 70% training, 30% validation
        X_train, X_val, y_train, y_val = split_nocv(X, y, test_size=0.3)

        # -------------- computing GG --------------
        compute_adjs(X_train, y_train, model_name, f"adjsnocv/train/{dataset}", params)

        # -------------- fitting classifier --------------
        clf = model(**params)
        clf.fit(X_train, y_train)

        # -------------- compute score --------------
        score = eval(X_val, y_val, n_classes, clf)

        return score

    else:
        """TBD"""
        return


def objective(trial, dataset, X, y, n_classes, model, parameters, CV):

    params = parameters(trial)

    print(model.__name__)

    if model.__name__ in ["Chipclass", "GGGMM", "GGRBF"] and params["reg"] is True:
        for c in np.arange(n_classes):
            params[f"perc{c + 1}"] = trial.suggest_float(f"perc{c + 1}", 50.0, 100.0)

    score = train_tune(dataset, X, y, n_classes, model, params, CV)

    return score


def call_eval(dataset, classifier, CV=False):

    # -------------- OPENML --------------------
    X, y = load(dataset)

    # -------------- GOING ON --------------------
    X = X.astype(float)
    y = y.astype(int)
    uniq_classes = np.unique(y)
    n_classes = len(uniq_classes)
    if n_classes == 2:
        assert 0 in uniq_classes and 1 in uniq_classes, "Check your labels"
    else:
        assert np.array_equal(np.arange(0, n_classes), uniq_classes), "Arrays are not equal"

    # -------------- dividing train/test -> test will remain untouched throughout fitting/hyperparameter tuning --------------
    X_train, X_test, y_train, y_test = split_nocv(X, y, test_size=0.2)
    # minmax scaling features
    X_train, X_test = normalize(X_train=X_train, X_test=X_test)

    assert classifier in list(CLFS.keys())
    clf = CLFS[classifier]["clf"]
    parameters_fun = CLFS[classifier]["params"]

    func = lambda trial: objective(trial, dataset, X_train, y_train, n_classes, clf, parameters_fun, CV)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=SEED))
    study.optimize(func, n_trials=N_ITER_OPTUNA)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:", study.best_trial)
    best_params = study.best_params
    if classifier in ["chipclass", "ggrbf", "gggmm"]:
        best_params["reg"] = True
        best_params["flex_reg"] = True
        compute_adjs(X_train, y_train, classifier, f"adjsnocv/train_n_val/{dataset}", best_params)

    if classifier == "svm":
        best_params["probability"] = True

    set_objective_tree_based(classifier, n_classes, best_params)

    # -------------- after choosing the best hyperparameters (on the training set), we retrain the model --------------
    clf = CLFS[classifier]["clf"]
    clf = clf(**best_params)
    clf.fit(X_train, y_train)

    # -------------- finally evaluating on the test set --------------
    score = eval(X_test, y_test, n_classes, clf)
    score = np.round(100 * score, 4)

    version = "1"
    file_dir = f"results/{version}/"

    Path(file_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{file_dir}/{classifier}.txt", "a+") as f:
        print(f"{dataset} & {score} ////", file=f)

    with open(f"{file_dir}/{classifier}_params.txt", "a+") as f:
        print(f"{dataset} & {best_params} ////", file=f)
