import numpy as np


def transf_rmv(df, y_col):
    """
    transform data to float and remove missing values
    """
    df = df.astype(float)
    df.dropna(inplace=True, axis=1, how='all')    # Remove columns where all elements are NaN
    df.dropna(inplace=True)
    df.drop_duplicates(keep='first', inplace=True)
    df.drop_duplicates(subset=list(df.columns[df.columns != y_col]), keep=False, inplace=True)

    return df


def get_arrays(df, y_col):
    X = np.array(df.loc[:, df.columns != y_col])
    y = np.array(df[y_col].tolist())
    return X, y
