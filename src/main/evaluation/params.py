import math


# resnet
def resnet(trial):
    """
    Based on:
    Revisiting Deep Learning Models for Tabular Data
    https://arxiv.org/abs/2106.11959

    hyperparameters based on table 14 of the paper:
    Table 14: ResNet hyperparameter space. Here (A) = {CA, AD, HE, JA, HI, AL} and
    (B) = {EP, YE, CO, YA, MI}
    Parameter (Datasets) Distribution
    # Layers (A) UniformInt[1, 8], (B) UniformInt[1, 16]
    Layer size (A) UniformInt[64, 512], (B) UniformInt[64, 1024]
    Hidden factor (A,B) Uniform[1, 4]
    Hidden dropout (A,B) Uniform[0, 0.5]
    Residual dropout (A,B) {0, Uniform[0, 0.5]}
    Learning rate (A,B) LogUniform[1e-5, 1e-2]
    Weight decay (A,B) {0, LogUniform[1e-6, 1e-3]}
    Category embedding size ({AD}) UniformInt[64, 512]
    # Iterations 100
    Hyperparameter space was chosen based on (A) = {CA, AD, HE, JA, HI, AL}.
    """
    param = {
        "n_blocks": trial.suggest_int("n_blocks", 1, 8, log=False),  # # Layers (A) UniformInt[1, 8],
        "d_block": trial.suggest_int("d_block", 64, 512, log=False),  # Layer size (A) UniformInt[64, 512],
        "d_hidden": trial.suggest_int("d_hidden", 1, 4, log=False),  # Hidden factor (A,B) Uniform[1, 4]
        "d_hidden_multiplier": trial.suggest_categorical(
            "d_hidden_multiplier", [None]
        ),  # will be None here since we'll already be using d_hidden
        "dropout1": trial.suggest_float("dropout1", 0, 0.5, log=False),  # Hidden dropout (A,B) Uniform[0, 0.5]
        "dropout2": trial.suggest_float("dropout2", 0, 0.5, log=False),  # Residual dropout (A,B) {0, Uniform[0, 0.5]}
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),  # Learning rate (A,B) LogUniform[1e-5, 1e-2]
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-6, 1e-3, log=True
        ),  # Weight decay (A,B) {0, LogUniform[1e-6, 1e-3]}
    }
    return param


# SVM
# based on tabpfn code
def svm(trial):
    param = {
        "probability": trial.suggest_categorical("probability", [True]),
        "C": trial.suggest_float("C", 1e-10, 1e10, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf"]),
        "gamma": trial.suggest_categorical("gamma", ["auto", "scale"]),
    }

    return param


# Random forest:
# Search space from
# https://www.kaggle.com/code/emanueleamcappella/random-forest-hyperparameters-tuning/notebook
# based also on tabfpn code
def random_forest(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 20, 200),
        "max_features": trial.suggest_categorical("max_features", ["log2", "sqrt"]),
        "max_depth": trial.suggest_int("max_depth", 1, 45),
        "min_samples_split": trial.suggest_categorical("min_samples_split", [5, 10]),
    }
    return param


# kNN:
# based on tabfpn code
def knn(trial):
    param = {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 16),
    }
    return param


# XGBoost
# Hyperparameter space: https://arxiv.org/pdf/2106.03253.pdf
# based also on tabfpn code
def xgboost(trial):
    param = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.2, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.2, 1),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-16, 100000, log=True),
        "alpha": trial.suggest_float("alpha", 1e-16, 100, log=True),
        "lambda": trial.suggest_float("lambda", 1e-16, 100, log=True),
        "gamma": trial.suggest_float("gamma", 1e-16, 100, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 4000),
    }
    return param


# Catboost
# Hyperparameter space: https://arxiv.org/pdf/2106.03253.pdf
# based also on tabfpn code
def catboost(trial):
    param = {
        "learning_rate": trial.suggest_float("learning_rate", math.exp(-5), 1, log=True),
        "random_strength": trial.suggest_int("random_strength", 1, 20),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1),
        "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 20),
        "iterations": trial.suggest_int("iterations", 100, 4000),
    }
    return param


# LGBM:
# based on tabfpn code
def lightgbm(trial):
    param = {
        "num_leaves": trial.suggest_int("num_leaves", 5, 50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 2000),
        "min_child_weight": trial.suggest_categorical(
            "min_child_weight", [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
        ),
        "subsample": trial.suggest_float("subsample", 0.2, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.8),
        "reg_alpha": trial.suggest_categorical("reg_alpha", [0, 1e-1, 1, 2, 5, 7, 10, 50, 100]),
        "reg_lambda": trial.suggest_categorical("reg_lambda", [0, 1e-1, 1, 5, 10, 20, 50, 100]),
    }
    return param


def ggrbf(trial):
    param = {
        "quality": trial.suggest_categorical("quality", ["dist_based"]),
        "sigma": trial.suggest_float("sigma", 1e-1, 10, log=True),
        "reg": trial.suggest_categorical("reg", [True]),
        "flex_reg": trial.suggest_categorical("flex_reg", [True]),
        "perc1": trial.suggest_float("perc1", 0, 100),
        "perc2": trial.suggest_float("perc2", 0, 100),
        "act_fun": trial.suggest_categorical("act_fun", ["tanhNorm"]),
    }

    return param


def ggrbficann(trial):
    param = {
        "quality": trial.suggest_categorical("quality", ["dist_based"]),
        "sigma": trial.suggest_float("sigma", 1e-1, 10, log=True),
        "reg": trial.suggest_categorical("reg", [True]),
        "flex_reg": trial.suggest_categorical("flex_reg", [True]),
        "perc1": trial.suggest_int("perc1", 10, 90),
        "perc2": trial.suggest_int("perc2", 10, 90),
        "act_fun": trial.suggest_categorical("act_fun", ["gaussian"]),
        "output_weights_algo": trial.suggest_categorical("output_weights_algo", ["pseudoinverse"]),
    }

    return param


def gmmgg(trial):
    param = {
        "quality": trial.suggest_categorical("quality", ["dist_based"]),
        "sigma": trial.suggest_float("sigma", 1e-1, 10, log=True),
        "reg": trial.suggest_categorical("reg", [True]),
        "flex_reg": trial.suggest_categorical("flex_reg", [True]),
        "perc1": trial.suggest_int("perc1", 10, 90),
        "perc2": trial.suggest_int("perc2", 10, 90),
        "act_fun": trial.suggest_categorical("act_fun", ["gaussian"]),
    }

    return param


def chipclass(trial):
    param = {
        "quality": trial.suggest_categorical("quality", ["dist_based"]),
        "sigma": trial.suggest_float("sigma", 1e-1, 10, log=True),
        "reg": trial.suggest_categorical("reg", [True]),
        "flex_reg": trial.suggest_categorical("flex_reg", [True]),
        "perc1": trial.suggest_int("perc1", 10, 90),
        "perc2": trial.suggest_int("perc2", 10, 90),
        "act_fun": trial.suggest_categorical("act_fun", ["tanhNorm"]),
    }

    return param


params = {
    "resnet": resnet,
    "svm": svm,
    "random_forest": random_forest,
    "knn": knn,
    "xgboost": xgboost,
    "catboost": catboost,
    "lightgbm": lightgbm,
    "ggrbf": ggrbf,
    "ggrbficann": ggrbficann,
    "gmmgg": gmmgg,
    "chipclass": chipclass,
}
