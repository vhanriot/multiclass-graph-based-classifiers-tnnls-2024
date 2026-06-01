import numpy as np
import pandas as pd
from sklearn import preprocessing

from main.preprocessing.utils import get_arrays, transf_rmv


def abalone(base_path:str = "data"):
    file = "abalone.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None)

    df = pd.get_dummies(df, columns=[0])
    y = df.pop(8)
    df = df.join(y)
    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)

    y_col = df.shape[1] - 1
    df[y_col] = df[y_col] - 1

    # Calculate value counts for the column
    value_counts = df[y_col].value_counts()

    # Identify values with counts less than 5
    values_to_drop = value_counts[value_counts < 5].index

    # Drop rows with values less than 5
    df = df[~df[y_col].isin(values_to_drop)]
    df.reset_index(inplace=True, drop=True)

    unique_values = df[y_col].unique()
    value_to_index = {value: index for index, value in enumerate(unique_values)}

    transformed_column = df[y_col].map(value_to_index)
    df[y_col] = transformed_column

    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)


def abalone_18vs9(base_path:str = "data"):
    file = "abalone.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None)

    df = pd.get_dummies(df, columns=[0])
    y = df.pop(8)
    df = df.join(y)
    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)

    y_col = df.shape[1] - 1
    df = df.loc[np.array(np.array(df.iloc[:, y_col]) == 18) | np.array(np.array(df.iloc[:, y_col]) == 9)]
    df.loc[np.array(df.iloc[:, y_col]) == 9, y_col] = -1
    df.loc[np.array(df.iloc[:, y_col]) == 18, y_col] = 1

    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)


def abalone_19vsall(base_path:str = "data"):
    file = "abalone.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None)

    df = pd.get_dummies(df, columns=[0])
    y = df.pop(8)
    df = df.join(y)
    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)
    y_col = df.shape[1] - 1
    df.loc[np.array(df.iloc[:, y_col]) != 19, y_col] = -1
    df.loc[np.array(df.iloc[:, y_col]) == 19, y_col] = 1

    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)


def appendicitis(base_path:str = "data"):
    file = "appendicitis.dat"

    df = pd.read_csv(f"{base_path}/{file}", skiprows=12, header=None)
    y_col = df.shape[1] - 1
    df.loc[np.array(df.iloc[:, y_col]) == 0, y_col] = -1
    df = df.astype(float)
    df.dropna(inplace=True)
    df.drop_duplicates(keep="first", inplace=True)
    df.drop_duplicates(subset=list(df.columns)[:-1], keep=False, inplace=True)

    return get_arrays(df, y_col)


def australian(base_path:str = "data"):
    file = "australian.dat"

    df = pd.read_csv(f"{base_path}/{file}", header=None, sep=" ")
    y_col = df.shape[1] - 1
    df.loc[np.array(df.iloc[:, y_col]) == 0, y_col] = -1
    df = df.astype(float)
    df.dropna(inplace=True)
    df.drop_duplicates(keep="first", inplace=True)
    df.drop_duplicates(subset=list(df.columns)[0:-1], keep=False, inplace=True)

    return get_arrays(df, y_col)


def banknote(base_path:str = "data"):
    file = "data_banknote_authentication.txt"

    df = pd.read_csv(f"{base_path}/{file}", header=None)
    y_col = df.shape[1] - 1
    df.loc[np.array(df.iloc[:, y_col]) == 0, y_col] = -1
    df = df.astype(float)
    df.dropna(inplace=True)
    df.drop_duplicates(keep="first", inplace=True)
    df.drop_duplicates(subset=list(df.columns)[0:-1], keep=False, inplace=True)

    return get_arrays(df, y_col)


def breastcancer_original(base_path:str = "data"):
    file = "breast-cancer-wisconsin.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None)
    df.drop([0], axis=1, inplace=True)
    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)
    y_col = df.shape[1] - 1
    df.loc[np.array(df.iloc[:, y_col]) == 2, y_col] = -1
    df.loc[np.array(df.iloc[:, y_col]) == 4, y_col] = 1
    df = df.replace({"?": None})
    df = df.astype(float)
    df.dropna(inplace=True)
    df.drop_duplicates(keep="first", inplace=True)
    df.drop_duplicates(subset=list(df.columns)[0:-1], keep=False, inplace=True)

    return get_arrays(df, y_col)


def breastcancer_prognostic(base_path:str = "data"):
    file = "wpbc.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None)
    df.drop([0, 2], axis=1, inplace=True)
    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)
    y_col = 0
    col = df.columns[y_col]
    df[col] = df[col].astype(object)
    df.loc[df[col] == "R", col] = -1
    df.loc[df[col] == "N", col] = 1
    df = df.replace({"?": None})
    df = df.astype(float)
    df.dropna(inplace=True)

    y = df.pop(y_col)
    df = df.join(y)
    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)
    df.drop_duplicates(keep="first", inplace=True)
    df.drop_duplicates(subset=list(df.columns)[0:-1], keep=False, inplace=True)

    return get_arrays(df, df.shape[1] - 1)


def car(base_path:str = "data"):
    file = "car.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None)
    df = pd.get_dummies(df, columns=[0, 1, 2, 3, 4, 5])
    y = df.pop(6)
    df = df.join(y)
    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)

    y_col = df.shape[1] - 1

    le = preprocessing.LabelEncoder()
    df[y_col] = le.fit_transform(df[y_col])

    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)


def climate(base_path:str = "data"):
    file = "pop_failures.dat"

    df = pd.read_csv(f"{base_path}/{file}", header=None, sep=r"\s+", skiprows=1)
    df.drop([0, 1], axis=1, inplace=True)
    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)
    y_col = df.shape[1] - 1
    df.loc[np.array(df.iloc[:, y_col]) == 0, y_col] = -1
    df = df.astype(float)
    df.dropna(inplace=True)
    df.drop_duplicates(keep="first", inplace=True)
    df.drop_duplicates(subset=list(df.columns)[0:-1], keep=False, inplace=True)

    return get_arrays(df, y_col)


def dna(base_path:str = "data"):
    file = "dna.arff"

    df = pd.read_csv(f"{base_path}/{file}", header=None, skiprows=188)

    y_col = df.shape[1] - 1
    df[y_col] = df[y_col] - 1

    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)


def fertility(base_path:str = "data"):
    file = "fertility_Diagnosis.txt"

    df = pd.read_csv(f"{base_path}/{file}", header=None)
    y_col = df.shape[1] - 1
    col = df.columns[y_col]
    df[col] = df[col].astype(object)
    df.loc[df[col] == "O", col] = -1
    df.loc[df[col] == "N", col] = 1
    df = df.astype(float)
    df.dropna(inplace=True)
    df.drop_duplicates(keep="first", inplace=True)
    df.drop_duplicates(subset=list(df.columns)[0:-1], keep=False, inplace=True)

    return get_arrays(df, y_col)


def glass_identification(base_path:str = "data"):
    file = "glass.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None, usecols=range(1, 11))
    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)

    y_col = df.shape[1] - 1
    df[y_col] = df[y_col] - 1
    df.loc[df[y_col] > 2, y_col] = df.loc[df[y_col] > 2, y_col] - 1

    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)


def glass_7vsall(base_path:str = "data"):
    file = "glass.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None, usecols=range(1, 11))
    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)

    y_col = df.shape[1] - 1
    df.loc[np.array(df.iloc[:, y_col]) != 7, y_col] = -1
    df.loc[np.array(df.iloc[:, y_col]) == 7, y_col] = 1

    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)


def haberman(base_path:str = "data"):
    file = "haberman.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None)
    y_col = df.shape[1] - 1
    df.loc[np.array(df.iloc[:, y_col]) == 2, y_col] = -1
    df = df.astype(float)
    df.dropna(inplace=True)
    df.drop_duplicates(keep="first", inplace=True)
    df.drop_duplicates(subset=list(df.columns)[0:-1], keep=False, inplace=True)

    return get_arrays(df, y_col)


def heart(base_path:str = "data"):
    file = "heart.dat"

    df = pd.read_csv(f"{base_path}/{file}", header=None, sep=r"\s+")
    y_col = df.shape[1] - 1
    df.loc[np.array(df.iloc[:, y_col]) == 2, y_col] = -1
    df = df.astype(float)
    df.dropna(inplace=True)
    df.drop_duplicates(keep="first", inplace=True)
    df.drop_duplicates(subset=list(df.columns)[0:-1], keep=False, inplace=True)

    return get_arrays(df, y_col)


def ilpd(base_path:str = "data"):
    file = "Indian Liver Patient Dataset (ILPD).csv"

    df = pd.read_csv(f"{base_path}/{file}", header=None)
    y_col = df.shape[1] - 1
    col = df.columns[y_col]
    df[col] = df[col].astype(object)
    df[1] = df[1].astype(object)
    df.loc[df[col] == 2, col] = -1
    df.loc[df[1] == "Female", 1] = 1
    df.loc[df[1] == "Male", 1] = 0
    df = df.astype(float)
    df.dropna(inplace=True)
    df.drop_duplicates(keep="first", inplace=True)
    df.drop_duplicates(subset=list(df.columns)[0:-1], keep=False, inplace=True)

    return get_arrays(df, y_col)


def ionosphere(base_path:str = "data"):
    file = "ionosphere.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None)
    y_col = df.shape[1] - 1
    col = df.columns[y_col]
    df[col] = df[col].astype(object)
    df.loc[df[col] == "b", col] = -1
    df.loc[df[col] == "g", col] = 1
    df = df.astype(float)
    df.dropna(inplace=True)
    df.drop_duplicates(keep="first", inplace=True)
    df.drop_duplicates(subset=list(df.columns)[0:-1], keep=False, inplace=True)

    return get_arrays(df, y_col)


def iris(base_path:str = "data"):
    file = "iris.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None)

    y_col = df.shape[1] - 1
    df.loc[df[y_col] == "Iris-setosa", y_col] = 0
    df.loc[df[y_col] == "Iris-versicolor", y_col] = 1
    df.loc[df[y_col] == "Iris-virginica", y_col] = 2

    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)


def parkinsons(base_path:str = "data"):
    file = "parkinsons.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None, skiprows=1)
    df.drop(0, axis=1, inplace=True)
    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)
    y_col = 16
    df.loc[np.array(df.iloc[:, y_col]) == 0, y_col] = -1
    df = df.astype(float)
    df.dropna(inplace=True)

    y = df.pop(y_col)
    df = df.join(y)
    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)
    df.drop_duplicates(keep="first", inplace=True)
    df.drop_duplicates(subset=list(df.columns)[0:-1], keep=False, inplace=True)

    return get_arrays(df, df.shape[1] - 1)


def satimage(base_path:str = "data"):
    file = "sat.trn"

    df = pd.read_csv(f"{base_path}/{file}", header=None, sep=r"\s+")
    y_col = df.shape[1] - 1
    df[y_col] = df[y_col] - 1
    df.loc[df[y_col] == 6, y_col] = 5

    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)


def seeds(base_path:str = "data"):
    file = "seeds_dataset.txt"

    df = pd.read_csv(f"{base_path}/{file}", header=None, sep=r"\s+")

    y_col = df.shape[1] - 1
    df[y_col] = df[y_col] - 1

    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)


def segmentation(base_path:str = "data"):
    file = "segmentation.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None, skiprows=5)

    y = df.pop(0)
    df = df.join(y)
    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)

    y_col = df.shape[1] - 1

    le = preprocessing.LabelEncoder()
    df[y_col] = le.fit_transform(df[y_col])

    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)


def vehicle(base_path:str = "data"):
    files = ["xaa.dat", "xab.dat", "xac.dat", "xad.dat", "xae.dat", "xaf.dat", "xag.dat", "xah.dat", "xai.dat"]
    df = pd.DataFrame()
    for file in files:
        df = pd.concat((df, pd.read_csv(f"{base_path}/{file}", header=None, sep=r"\s+")), ignore_index=True)

    y_col = df.shape[1] - 1
    df.loc[np.array(df.iloc[:, y_col]) == "saab", y_col] = 0
    df.loc[np.array(df.iloc[:, y_col]) == "van", y_col] = 1
    df.loc[np.array(df.iloc[:, y_col]) == "bus", y_col] = 2
    df.loc[np.array(df.iloc[:, y_col]) == "opel", y_col] = 3

    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)


def vehicle_vanvsall(base_path:str = "data"):
    files = ["xaa.dat", "xab.dat", "xac.dat", "xad.dat", "xae.dat", "xaf.dat", "xag.dat", "xah.dat", "xai.dat"]
    df = pd.DataFrame()
    for file in files:
        df = pd.concat((df, pd.read_csv(f"{base_path}/{file}", header=None, sep=r"\s+")), ignore_index=True)

    y_col = df.shape[1] - 1
    col = df.columns[y_col]
    df[col] = df[col].astype(object)
    df.loc[df[col] != "van", col] = -1
    df.loc[df[col] == "van", col] = 1
    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)


def vowel(base_path:str = "data"):
    file = "vowel-context.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None, sep=r"\s+")
    df.pop(0)
    df.pop(1)
    df.pop(2)
    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)

    y_col = df.shape[1] - 1
    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)


def wine(base_path:str = "data"):
    file = "wine.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None)

    y = df.pop(0)
    df = df.join(y)
    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)

    y_col = df.shape[1] - 1
    df[y_col] = df[y_col] - 1

    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)


def yeast(base_path:str = "data"):
    file = "yeast.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None, sep=r"\s+", usecols=range(1, 10))

    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)

    y_col = df.shape[1] - 1

    le = preprocessing.LabelEncoder()
    df[y_col] = le.fit_transform(df[y_col])

    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)


def yeast_9vs1(base_path:str = "data"):
    file = "yeast.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None, sep=r"\s+", usecols=range(1, 10))
    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)

    y_col = df.shape[1] - 1
    df = df.loc[np.array(np.array(df.iloc[:, y_col]) == "POX") | np.array(np.array(df.iloc[:, y_col]) == "CYT")]
    col = df.columns[y_col]
    df[col] = df[col].astype(object)
    df.loc[df[col] == "CYT", col] = -1
    df.loc[df[col] == "POX", col] = 1

    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)


def yeast_5vsall(base_path:str = "data"):
    file = "yeast.data"

    df = pd.read_csv(f"{base_path}/{file}", header=None, sep=r"\s+", usecols=range(1, 10))
    df.columns = np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)

    y_col = df.shape[1] - 1
    col = df.columns[y_col]
    df[col] = df[col].astype(object)
    df.loc[df[col] != "ME2", col] = -1
    df.loc[df[col] == "ME2", col] = 1

    df = transf_rmv(df, y_col)

    return get_arrays(df, y_col)
