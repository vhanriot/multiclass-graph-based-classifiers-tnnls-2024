import numpy as np

from main.models.graph_models.Chipclass import Chipclass


class GGGMM(Chipclass):
    r"""Gabriel Graph-based Gaussian Mixtures Models

    Implementation of the classifier presented in:

    Torres, L. C., Castro, C. L., Coelho, F., & Braga, A. P. (2020). Large margin gaussian mixture classifier with a gabriel graph geometric representation of data set structure. IEEE Transactions on Neural Networks and Learning Systems, 32(3), 1400-1406.
    - DOI: 10.1109/TNNLS.2020.2980559
    - Link: https://ieeexplore.ieee.org/abstract/document/9064693
    - BibTeX citation:
        @article{torres2020large,
        title={Large margin gaussian mixture classifier with a gabriel graph geometric representation of data set structure},
        author={Torres, Luiz CB and Castro, Cristiano L and Coelho, Frederico and Braga, Ant{\^o}nio P},
        journal={IEEE Transactions on Neural Networks and Learning Systems},
        volume={32},
        number={3},
        pages={1400--1406},
        year={2020},
        publisher={IEEE}
        }

    Parameters:
    ----------
    quality : str, mandatory (default=cardinality)
        Function used to compute the membership value of each training sample.

    reg : bool, mandatory (default=True)
        Whether to apply regularization or not.

    flex_reg : bool, mandatory (default=False)
        Whether to apply flexible regularization or not.

    act_fun : str, mandatory (default=gaussian)
        Activation function of the hidden layer.

    sigma : float, optional (default=1e-1)
        Sigma value of the gaussian kernel used when quality='dist_based'.

    D : np.ndarray, optional (default=None)
        Distance matrix of the training set.

    adj : np.ndarray, optional (default=None)
        Adjacency matrix of the training set.

    adj_in : np.ndarray, optional (default=None)
        "Within" adjacency matrix of the training set.

    Attributes:
    ----------
    trained : bool
        True if the model has been trained, False otherwise.

    classes : np.ndarray
        Unique labels of the training set.

    n_classes : int
        Number of classes of the training set.

    D : np.ndarray
        Distance matrix of the training set (whether or not filtering is applied).

    adj : np.ndarray
        Adjacency matrix of the training set (whether or not filtering is applied).

    adj_in : np.ndarray
        "Within" adjacency matrix of the training set (whether or not filtering is applied).

    percentages : np.ndarray
        Percentage of the number of samples to be removed from each class.

    ssvs : np.ndarray
        Structural support vectors.

    y_ssvs : np.ndarray
        Structural support vectors' labels.

    sigmas : np.ndarray
        Gaussian mixture parameters.

    prioris : list
        Prior probabilities of each class.

    Example:
    ----------
        >>> from sklearn import datasets
        >>> from sklearn.model_selection import train_test_split
        >>> dataset = datasets.load_iris()
        >>> X = dataset["data"]
        >>> y = dataset["target"]
        >>> y = y.astype(float)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        >>> # implementation of Torres, Luiz CB, et al. with fixed regularization
        >>> clf = GGGMM()
        >>> clf.fit(X_train, y_train)
        >>> yhat = clf.predict(X_test)
        >>> # implementation of Torres, Luiz CB, et al. with flexible regularization of Hanriot, Vítor M, et al.
        >>> clf = GGGMM(flex_reg=True, perc1=20, perc2=30, perc3=40)
        >>> clf.fit(X_train, y_train)
        >>> yhat = clf.predict(X_test)
    """

    def __init__(
        self,
        quality="cardinality",
        reg=True,
        flex_reg=False,
        act_fun="gaussian",
        sigma=None,
        D=None,
        adj=None,
        adj_in=None,
        **kwargs,
    ) -> None:
        """Initialize the model (same hyperparameters as Chipclass, but some different default values)."""
        super().__init__(quality, reg, flex_reg, act_fun, sigma, D, adj, adj_in, **kwargs)

    def _compute_params(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute gaussian mixture parameters (kernels' sigmas) following Eqs. 8, 9 and 10 of the paper.
        Stores structural support vectors in self.ssvs, as well as their labels in self.y_ssvs.

        Args:
        X (array-like): Training data of shape (n_samples, n_features).
        y (array-like): Target values of shape (n_samples,).
        """
        self.ssvs = X[self.ssv.bound_pairs.reshape(-1)]
        self.y_ssvs = y[self.ssv.bound_pairs.reshape(-1)]
        # _D = np.sqrt(self.D)
        self.sigmas = []
        for bound_pair in self.ssv.bound_pairs:
            distance_pair = self._dist(X[bound_pair[0]].reshape(1, -1), X[bound_pair[1]].reshape(1, -1))[0][0]
            self.sigmas.append((distance_pair / 2) / 3)
        self.sigmas = np.vstack((self.sigmas, self.sigmas)).T.reshape(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train GGGMM: compute gaussian mixture parameters and prior probabilities."""
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)

        if self.n_classes < 2:
            raise Exception("Less than 2 classes")

        # compute the gabriel graph for the first time (if it's None)
        if self.adj is None:
            self._gg(X)

        if self.reg == False:
            # find ssvs
            self.ssv.fit(self.adj, y)
            # store ssvs and compute sigmas
            self._compute_params(X, y)
        else:
            # compute membership function (for filtering procedure)
            self.ssv._get_all_neighbors(self.adj)
            quality = self._quality.forward(self.ssv.all_neighbors, y, self.D)
            # filtering procedure
            self._build_perc_thresholds()
            self.valid_samples = self.filter.forward(quality, y, self.percentages)
            valid_index = np.where(np.array(self.valid_samples) == 1)[0]
            # recompute gg
            X = X[valid_index]
            y = y[valid_index]
            si = np.where(np.array(self.valid_samples) == 0)[0]
            self._subgg(si)
            # find new ssvs
            self.ssv.fit(self.adj, y)
            # store ssvs and compute sigmas
            self._compute_params(X, y)

        self.prioris = [np.sum(y == c) / len(y) for c in self.classes_]  # used algo 1 paper line 34
        self.trained = True

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward-pass through the NN."""
        d = X.shape[1]
        norm = 1 / (pow(2 * np.pi, d / 2) * pow(np.array(self.sigmas), 2 * d))
        H = norm * self._act_fun.forward(self._dist(X, self.ssvs), self.sigmas)
        posterioris = np.zeros((H.shape[0], self.n_classes))
        for c in range(self.n_classes):
            posterioris[:, c] = self.prioris[c] * np.sum(H[:, self.y_ssvs == c], axis=1)
        posterioris = posterioris / posterioris.sum(axis=1, keepdims=True)
        return posterioris

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns the probability of each class for each sample of the test set."""
        return self._forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns the predicted class for each sample of the test set."""
        posterioris = self._forward(X)
        return np.argmax(posterioris, axis=1)
