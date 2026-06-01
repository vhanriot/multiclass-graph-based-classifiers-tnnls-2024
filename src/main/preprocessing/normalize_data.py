from sklearn.preprocessing import MinMaxScaler


def normalize(X_train, X_test=None):
    """
    Normalize the features using Min-Max scaling.

    Parameters:
    - X_train (numpy array): Training data.
    - X_test (numpy array, optional): Test data. Default is None.

    Returns:
    - X_train_norm (numpy array): Normalized training data.
    - X_test_norm (numpy array or None): Normalized test data if provided, otherwise None.
    """

    scaler = MinMaxScaler()

    X_train_norm = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_norm = scaler.transform(X_test)
    else:
        X_test_norm = None

    return X_train_norm, X_test_norm
