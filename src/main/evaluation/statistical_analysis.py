import numpy as np
import pandas as pd


# https://www.kaggle.com/code/pbizil/machine-learning-models-and-friedman-test/notebook
def ranking_model(resultsagg: pd.DataFrame) -> pd.DataFrame:
    ranking = pd.DataFrame(columns=resultsagg.columns)
    for i in range(resultsagg.shape[0]):
        ranking.loc[i, resultsagg.iloc[i].rank(ascending=False).index] = resultsagg.iloc[i].rank(ascending=False)
    return ranking


def calc_Xf2(N: int, k: int, R: np.ndarray) -> float:
    first_term = 12 * N / (k * (k + 1))
    second_term = np.sum(R**2) - (k * (k + 1) ** 2) / 4
    xf2: float = first_term * second_term
    return xf2


def calc_Ff(N: int, k: int, Xf2: float) -> float:
    return (N - 1) * Xf2 / (N * (k - 1) - Xf2)


def calc_CD(k: int, N: int, qa: float) -> float:
    cd: float = qa * np.sqrt((k * (k + 1)) / (6 * N))
    return cd


# Demšar, Janez. "Statistical comparisons of classifiers over multiple data sets." The Journal of Machine learning research 7 (2006): 1-30.
# table 5b, q0.05 (prob 0.95)
qa = {"2": 1.960, "3": 2.241, "4": 2.394, "5": 2.498, "6": 2.576, "7": 2.638, "8": 2.690, "9": 2.724, "10": 2.773}

alias = {
    "abalone_18vs9": "a18-9",
    "abalone_19vsall": "a19",
    "appendicitis": "apd",
    "australian": "aust",
    "banknote": "bnk",
    "breastcancer_original": "bco",
    "breastcancer_prognostic": "bcp",
    "climate": "cli",
    "fertility": "fer",
    "glass_7vsall": "gls7",
    "haberman": "hab",
    "heart": "heart",
    "ilpd": "ilpd",
    "ionosphere": "iono",
    "parkinsons": "par",
    "vehicle_vanvsall": "v4",
    "yeast_5vsall": "yst5",
    "yeast_9vs1": "yst9-1",
    "car": "car",
    "glass_identification": "glass",
    "iris": "iris",
    "satimage": "sat",
    "seeds": "sds",
    "segmentation": "seg",
    "vehicle": "veh",
    "vowel": "vow",
    "wine": "wine",
    "yeast": "yst",
}
