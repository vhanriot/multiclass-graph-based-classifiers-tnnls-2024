import numpy as np
from sklearn.base import BaseEstimator

from main.models.graph_models.graph.ActFun import ActFun
from main.models.graph_models.graph.Distance import Distance
from main.models.graph_models.graph.Filter import Filter
from main.models.graph_models.graph.GG import GG
from main.models.graph_models.graph.Quality import Quality
from main.models.graph_models.graph.SSV import SSV


class Chipclass(BaseEstimator):
    r"""Chipclass

    Implementation of the classifier presented in:

    Torres, L. C. B., Castro, C. L., Coelho, F., Sill Torres, F., & Braga, A. P. (2015). Distance‐based large margin classifier suitable for integrated circuit implementation. Electronics Letters, 51(24), 1967-1969.
    - DOI: 10.1049/el.2015.1644
    - Link: https://doi.org/10.1049/el.2015.1644
    - BibTeX citation:
        @article{torres2015distance,
        title={Distance-based large margin classifier suitable for integrated circuit implementation},
        author={Torres, LCB and Castro, CL and Coelho, F and Sill Torres, F and Braga, AP},
        journal={Electronics Letters},
        volume={51},
        number={24},
        pages={1967--1969},
        year={2015},
        publisher={Wiley Online Library}
        }

    Most recent Chipclass papers on embedded systems:

    Arias-Garcia, J., de Souza, A. C., Gade, L., Yudi, J., Coelho, F., Castro, C. L., ... & Braga, A. P. (2022). Improved Design for Hardware Implementation of Graph-Based Large Margin Classifiers for Embedded Edge Computing. IEEE Transactions on Neural Networks and Learning Systems.
    - DOI: 10.1109/TNNLS.2022.3183236
    - Link: https://ieeexplore.ieee.org/abstract/document/9805692
    - BibTeX citation:
        @article{arias2022improved,
        title={Improved Design for Hardware Implementation of Graph-Based Large Margin Classifiers for Embedded Edge Computing},
        author={Arias-Garcia, Janier and de Souza, Alan C{\^a}ndido and Gade, Liliane and Yudi, Jones and Coelho, Frederico and Castro, Cristiano L and Torres, Luiz CB and Braga, Antonio P},
        journal={IEEE Transactions on Neural Networks and Learning Systems},
        year={2022},
        publisher={IEEE}
        }

    Arias-Garcia, J., Mafra, A., Gade, L., Coelho, F., Castro, C., Torres, L., & Braga, A. (2020). Enhancing performance of gabriel graph-based classifiers by a hardware co-processor for embedded system applications. IEEE Transactions on Industrial Informatics, 17(2), 1186-1196.
    - DOI: 10.1109/TII.2020.2987329
    - Link: https://ieeexplore.ieee.org/abstract/document/9072429
    - BibTeX citation:
        @article{arias2020enhancing,
        title={Enhancing performance of gabriel graph-based classifiers by a hardware co-processor for embedded system applications},
        author={Arias-Garcia, Janier and Mafra, Augusto and Gade, Liliane and Coelho, Frederico and Castro, Cristiano and Torres, Luiz and Braga, Ant{\^o}nio},
        journal={IEEE Transactions on Industrial Informatics},
        volume={17},
        number={2},
        pages={1186--1196},
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

    act_fun : str, mandatory (default=tanh)
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

    ssv_pos : np.ndarray
        Structural support vectors of the positive class.

    ssv_neg : np.ndarray
        Structural support vectors of the negative class.

    midpoints: np.ndarray
        Midpoints of each SSV-pair.

    Example:
    ----------
        >>> from sklearn import datasets
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import MinMaxScaler
        >>> dataset = datasets.load_breast_cancer()
        >>> X = dataset["data"]
        >>> y = dataset["target"]
        >>> y = y.astype(float)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        >>> scaler = MinMaxScaler()
        >>> X_train = scaler.fit_transform(X_train)
        >>> X_test = scaler.transform(X_test)
        >>> # implementation of Torres, Luiz CB, et al. with fixed regularization
        >>> clf = Chipclass()
        >>> clf.fit(X_train, y_train)
        >>> yhat = clf.predict(X_test)
        >>> # implementation of Torres, Luiz CB, et al. with flexible regularization of Hanriot, Vítor M, et al.
        >>> clf = Chipclass(flex_reg=True, perc1=20, perc2=30)
        >>> clf.fit(X_train, y_train)
        >>> yhat = clf.predict(X_test)
    """

    def __init__(
        self,
        quality="cardinality",
        reg=True,
        flex_reg=False,
        act_fun="tanh",
        sigma=1e-1,
        D=None,
        adj=None,
        adj_in=None,
        **kwargs,
    ) -> None:
        """Initialize the model."""
        self.quality = quality
        self.reg = reg
        self.flex_reg = flex_reg
        self.act_fun = act_fun
        self.sigma = sigma
        self.D = D
        self.adj = adj
        self.adj_in = adj_in
        self.gg = GG()
        self.ssv = SSV()
        self.filter = Filter(self.flex_reg)
        self._act_fun = ActFun(self.act_fun)
        self._quality = Quality(self.quality, self.sigma)
        self.dist = Distance(operation="scipy", squared=False, zero_dist_to_inf=False)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._param_names = ["quality"] + ["reg"] + ["flex_reg"] + ["act_fun"] + ["sigma"] + list(kwargs.keys())

    def get_params(self, deep=True):
        """Make hyperparameters available at self.get_params() (based on scikit-learn's BaseEstimator)."""
        return {param: getattr(self, param) for param in self._param_names}

    def set_params(self, **parameters):
        """Let user set a new hyperparameter (based on scikit-learn's BaseEstimator)."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            if parameter not in self._param_names:
                self._param_names.append(parameter)
        return self

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def _dist(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Computes the Euclidian distance matrix between matrices A and B."""
        return self.dist.forward(A, B)

    def _build_perc_thresholds(self) -> None:
        """Based on kwargs: "perc" hyperparameter passed in __init___().

        Example:
        ----------
            >>> # Binary classification problem:
            >>> clf = self(flex_reg=True, perc1=30, perc2=20)
            >>> clf.Chipclass(X_train, y_train)
            >>> clf.percentages
            array([30,20])
            >>> # 3-class classification problem:
            >>> clf = Chipclass(flex_reg=True, perc1=10.25, perc2=50.1, perc3=5.2)
            >>> clf.fit(
            ...     X_train, y_train
            ... )  # this will actually return an error for Chipclass since it hasn't been implemented for multi-class classification problems yet; but will work for other classifiers such as GGRBF and GGGMM
            >>> clf.percentages
            array([10.25, 50.1, 5.2])
        """
        arr_keys = sorted([k for k in self.__dict__.keys() if k.startswith("perc")], key=lambda x: int(x[4:]))
        self.percentages = np.array([self.__dict__[k] for k in arr_keys])

    def _gg(self, X: np.ndarray) -> None:
        """Computes the Gabriel graph."""
        self.D, self.adj, self.adj_in = self.gg.vectorized_in(X)

    def _subgg(self, si) -> None:
        """Recomputes the Gabriel graph."""
        self.adj = self.gg.sub_GG(self.adj_in, self.D, si)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train Chipclass: compute midpoints and positive and negative structural support vectors."""
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)

        if self.n_classes < 2:
            raise Exception("Less than 2 classes")
        elif self.n_classes > 2:
            raise Exception("Classifier not (yet) implemented for multi-class problems (should try ovo and ova)")

        # compute the gabriel graph for the first time (if it's None)
        if self.adj is None:
            self._gg(X)

        if self.reg == False:
            # find ssvs
            self.ssv.fit(self.adj, y)
            # store midpoints and -/+ ssvs
            self.midpoints = (X[self.ssv.bound_pairs[:, 0]] + X[self.ssv.bound_pairs[:, 1]]) / 2
            Y = y.reshape(-1, 1)
            self.ssv_neg = np.where(
                Y[self.ssv.bound_pairs[:, 0]] == 0, X[self.ssv.bound_pairs[:, 0]], X[self.ssv.bound_pairs[:, 1]]
            )
            self.ssv_pos = np.where(
                Y[self.ssv.bound_pairs[:, 0]] == 0, X[self.ssv.bound_pairs[:, 1]], X[self.ssv.bound_pairs[:, 0]]
            )
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
            # store midpoints and -/+ ssvs
            self.midpoints = (X[self.ssv.bound_pairs[:, 0]] + X[self.ssv.bound_pairs[:, 1]]) / 2
            Y = y.reshape(-1, 1)
            self.ssv_neg = np.where(
                Y[self.ssv.bound_pairs[:, 0]] == 0, X[self.ssv.bound_pairs[:, 0]], X[self.ssv.bound_pairs[:, 1]]
            )
            self.ssv_pos = np.where(
                Y[self.ssv.bound_pairs[:, 0]] == 0, X[self.ssv.bound_pairs[:, 1]], X[self.ssv.bound_pairs[:, 0]]
            )
        self.trained = True

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward-pass through the NN."""
        H = self._act_fun.forward(self._dist(X, self.midpoints))
        D_neg = self._dist(X, self.ssv_neg)
        D_pos = self._dist(X, self.ssv_pos)
        W = np.sign(1 * (D_pos - D_neg <= 0) - 0.5)
        return np.sum(np.multiply(H, W), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns the probability of each class for each sample of the test set."""
        yhat = self._forward(X)
        P_pos = self._sigmoid(yhat)
        P_pos = P_pos.reshape((len(P_pos), 1))
        P_neg = 1 - P_pos
        return np.concatenate((P_neg, P_pos), axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns the predicted class for each sample of the test set."""
        yhat = self._forward(X)
        yhat = np.sign(yhat)
        yhat[yhat == -1] = 0
        return yhat
