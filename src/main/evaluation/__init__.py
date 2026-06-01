import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from main.models.graph_models.graph.GG import GG
from main.models.graph_models.graph.SSV import SSV
from main.preprocessing import load_data
from main.preprocessing.normalize_data import normalize
from main.preprocessing.split_data import split_nocv

ALL_DATASETS = {
    "appendicitis": load_data.appendicitis,
    "australian": load_data.australian,
    "banknote": load_data.banknote,
    "breastcancer_original": load_data.breastcancer_original,
    "breastcancer_prognostic": load_data.breastcancer_prognostic,
    "climate": load_data.climate,
    "fertility": load_data.fertility,
    "haberman": load_data.haberman,
    "heart": load_data.heart,
    "ilpd": load_data.ilpd,
    "ionosphere": load_data.ionosphere,
    "parkinsons": load_data.parkinsons,
    "glass_7vsall": load_data.glass_7vsall,
    "vehicle_vanvsall": load_data.vehicle_vanvsall,
    "yeast_5vsall": load_data.yeast_5vsall,
    "yeast_9vs1": load_data.yeast_9vs1,
    "abalone_18vs9": load_data.abalone_18vs9,
    "abalone_19vsall": load_data.abalone_19vsall,
    "abalone": load_data.abalone,
    "car": load_data.car,
    "glass_identification": load_data.glass_identification,
    "iris": load_data.iris,
    "satimage": load_data.satimage,
    "seeds": load_data.seeds,
    "segmentation": load_data.segmentation,
    "vehicle": load_data.vehicle,
    "vowel": load_data.vowel,
    "wine": load_data.wine,
    "yeast": load_data.yeast,
    "dna": load_data.dna,
}


def eval(X, y, n_classes, clf):
    if n_classes == 2:
        yhat = clf.predict_proba(X)[:, 1]
        score = roc_auc_score(y_true=y, y_score=yhat)
    else:
        yhat = clf.predict_proba(X)
        score = roc_auc_score(y_true=y, y_score=yhat, multi_class="ovo", average="macro")
    return score


def set_objective_tree_based(clf_name, n_classes, params):
    if clf_name == "XGBClassifier":
        if n_classes == 2:
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "auc"
        else:
            params["objective"] = "multi:softprob"
            params["num_class"] = n_classes
            params["eval_metric"] = "mlogloss"
    elif clf_name == "CatBoostClassifier":
        params["verbose"] = False
    elif clf_name == "LGBMClassifier":
        params["verbose"] = -1
        if n_classes == 2:
            params["objective"] = "binary"
            params["metric"] = "auc"
        else:
            params["objective"] = "multiclass"
            params["num_class"] = n_classes
            params["metric"] = "multiclass"


def place_centers(model_name, X, file_path, params, y=None):
    del y
    if model_name not in ["Chipclass", "GGGMM", "GGRBF"]:
        return

    filename_D = f"{file_path}/D.npy"
    filename_adj = f"{file_path}/adj.npy"
    filename_adj_in = f"{file_path}/adj_in.npy"

    if not os.path.exists(filename_D):
        print("computing GG")
        gg = GG()
        D, adj, adj_in = gg.vectorized_in(X)

        Path(file_path).mkdir(parents=True, exist_ok=True)
        np.save(filename_D, np.array(D))
        np.save(filename_adj, np.array(adj))
        np.save(filename_adj_in, np.array(adj_in))
    else:
        D = np.load(filename_D)
        adj = np.load(filename_adj)
        adj_in = np.load(filename_adj_in)

    params["D"] = D
    params["adj"] = adj
    params["adj_in"] = adj_in


__all__ = [
    "ALL_DATASETS",
    "GG",
    "SSV",
    "eval",
    "normalize",
    "np",
    "place_centers",
    "set_objective_tree_based",
    "split_nocv",
]
