import openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from main.preprocessing.utils import get_arrays, transf_rmv


def load(openml_id, X=None, y=None):

    dataset = openml.datasets.get_dataset(
        openml_id, download_data=False, download_qualities=False, download_features_meta_data=False
    )
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    categorical_columns = X.columns[[i == True for i in categorical_indicator]]

    X = pd.get_dummies(X, columns=categorical_columns)
    X["y_htOw7yj66q5l"] = y
    label_encoder = LabelEncoder()
    X["y_htOw7yj66q5l"] = label_encoder.fit_transform(X["y_htOw7yj66q5l"])

    X = transf_rmv(X, y_col="y_htOw7yj66q5l")

    X, y = get_arrays(X, "y_htOw7yj66q5l")

    return X, y
