import math


# chipclass
def chipclass(trial):
    param = {
        "quality": trial.suggest_categorical("quality", ["cardinality", "dist_based"]),
        "reg": True,
        "flex_reg": True,
        "act_fun": trial.suggest_categorical("act_fun", ["chipclass", "tanhNorm"]),
    }

    if param["quality"] == "dist_based":
        param["sigma"] = trial.suggest_float("sigma", 1e-2, 1e-1)

    return param


# GGRBF
def ggrbf(trial):
    param = {
        "quality": trial.suggest_categorical("quality", ["cardinality", "dist_based"]),
        "reg": True,
        "flex_reg": True,
    }

    if param["quality"] == "dist_based":
        param["sigma"] = trial.suggest_float("sigma", 1e-2, 1e-1)

    return param


# GGGMM
def gggmm(trial):
    param = {
        "quality": trial.suggest_categorical("quality", ["cardinality", "dist_based"]),
        "reg": True,
        "flex_reg": True,
    }

    if param["quality"] == "dist_based":
        param["sigma"] = trial.suggest_float("sigma", 1e-2, 1e-1)

    return param


# SVM
def svm(trial):
    param = {
        "probability": True,
        "C": trial.suggest_float("C", .1, 1e3, log=True),
        "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    }

    if param["kernel"] in ["rbf", "sigmoid"]:
        param["gamma"] = trial.suggest_categorical("gamma", ["auto", "scale"])
    elif param["kernel"] == "poly":
        param['degree'] = trial.suggest_int("degree", 1, 10)

    return param


# Random forest:
# Search space from
# https://www.kaggle.com/code/emanueleamcappella/random-forest-hyperparameters-tuning/notebook
# based also on tabpfn code
def random_forest(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 20, 200),
        'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt']),
        'max_depth': trial.suggest_int('max_depth', 1, 45),
        'min_samples_split': trial.suggest_categorical('min_samples_split', [5, 10]),
    }
    return param


# kNN:
# based on tabpfn code
def knn(trial):
    param = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 16),
    }
    return param


# XGBoost
# Hyperparameter space: https://arxiv.org/pdf/2106.03253.pdf
# based also on tabpfn code
def xgboost(trial):
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-7, 1, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.2, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.2, 1),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-16, 100000, log=True),
        'alpha': trial.suggest_float('alpha', 1e-16, 100, log=True),
        'lambda': trial.suggest_float('lambda', 1e-16, 100, log=True),
        'gamma': trial.suggest_float('gamma', 1e-16, 100, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 4000),
    }
    return param


# Catboost
# Hyperparameter space: https://arxiv.org/pdf/2106.03253.pdf
# based also on tabpfn code
def catboost(trial):
    param = {
        'learning_rate': trial.suggest_float('learning_rate', math.exp(-5), 1, log=True),
        'random_strength': trial.suggest_int('random_strength', 1, 20),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0., 1),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 20),
        'iterations': trial.suggest_int('iterations', 100, 4000),
    }
    return param


# LGBM:
# based on tabpfn code
def lightgbm(trial):
    param = {
        'num_leaves': trial.suggest_int('num_leaves', 5, 50),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 2000),
        'min_child_weight': trial.suggest_categorical('min_child_weight', [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]),
        'subsample': trial.suggest_float('subsample', 0.2, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.8),
        'reg_alpha': trial.suggest_categorical('reg_alpha', [0, 1e-1, 1, 2, 5, 7, 10, 50, 100]),
        'reg_lambda': trial.suggest_categorical('reg_lambda', [0, 1e-1, 1, 5, 10, 20, 50, 100]),
    }
    return param


hyperparameters = {'chipclass': chipclass,
                   'ggrbf': ggrbf,
                   'gggmm': gggmm,
                   'svm': svm,
                   'random_forest': random_forest,
                   'knn': knn,
                   'xgboost': xgboost,
                   'catboost': catboost,
                   'lightgbm': lightgbm}
