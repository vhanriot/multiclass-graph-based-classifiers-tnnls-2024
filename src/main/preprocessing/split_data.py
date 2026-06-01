from sklearn.model_selection import StratifiedKFold, train_test_split


def split_nocv(X, y, test_size=0.2, random_state: int = 21):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def split_cv(n_splits: int, random_state: int = 21, shuffle: bool = True):
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    return skf
