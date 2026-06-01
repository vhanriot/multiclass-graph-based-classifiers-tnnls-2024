import lightning as L
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from main.models.base_models.CustomDataset import CustomDataset
from main.models.graph_models.Chipclass import Chipclass

torch.set_float32_matmul_precision("medium")


class MyLinearLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class LightningModel(L.LightningModule):
    def __init__(
        self,
        model,
        lr,
        weight_decay,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _common_step(self, batch):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def training_step(self, batch):
        # print('TRAINING')
        loss, scores, y = self._common_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        # print('VALIDATING')
        loss, scores, y = self._common_step(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch):
        # print('TESTING')
        loss, scores, y = self._common_step(batch)
        self.log("test_loss", loss)
        return loss


class GGRBF(Chipclass):
    """Gabriel Graph-based Radial Basis Function Network

    Implementation of the classifier presented in:

    Torres, L. C., Lemos, A. P., Castro, C. L., & Braga, A. P. (2014). A geometrical approach for parameter selection of radial basis functions networks. In Artificial Neural Networks and Machine Learning–ICANN 2014: 24th International Conference on Artificial Neural Networks, Hamburg, Germany, September 15-19, 2014. Proceedings 24 (pp. 531-538). Springer International Publishing.
    - DOI: 10.1007/978-3-319-11179-7_67
    - Link: https://link.springer.com/chapter/10.1007/978-3-319-11179-7_67
    - BibTeX citation:
        @inproceedings{torres2014geometrical,
        title={A geometrical approach for parameter selection of radial basis functions networks},
        author={Torres, Luiz CB and Lemos, Andr{\'e} P and Castro, Cristiano L and Braga, Ant{\\^o}nio P},
        booktitle={Artificial Neural Networks and Machine Learning--ICANN 2014: 24th International Conference on Artificial Neural Networks, Hamburg, Germany, September 15-19, 2014. Proceedings 24},
        pages={531--538},
        year={2014},
        organization={Springer}
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
        >>> clf = GGRBF()
        >>> clf.fit(X_train, y_train)
        >>> yhat = clf.predict(X_test)
        >>> # implementation of Torres, Luiz CB, et al. with flexible regularization of Hanriot, Vítor M, et al.
        >>> clf = GGRBF(flex_reg=True, perc1=20, perc2=30, perc3=40)
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
        output_weights_algo="pseudoinverse",
        lr=None,
        weight_decay=None,
        n_epochs=None,
        **kwargs,
    ) -> None:
        """Initialize the model (same hyperparameters as Chipclass, but some different default values)."""
        super().__init__(quality, reg, flex_reg, act_fun, sigma, D, adj, adj_in, **kwargs)
        self.output_weights_algo = output_weights_algo
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.precision = "bf16-mixed"
        self.batch_size = 32
        self.softmax = nn.Softmax(dim=1)

    def _compute_sigmas(self, y: np.ndarray) -> None:
        """Compute kernels' sigmas following Eq. 5 of the paper."""
        self.ssv_indexes = np.unique(self.ssv.bound_pairs)
        if self.act_fun == "gaussian":
            sigmas = []
            for ssv_index in self.ssv_indexes:
                ssv_neighbors = self.ssv.all_neighbors[ssv_index]
                ssv_opposite_neighbors = ssv_neighbors[y[ssv_neighbors] != y[ssv_index]]
                squared_dists = self.D[ssv_index, ssv_opposite_neighbors]
                min_dist = np.sqrt(np.min(squared_dists))
                sigmas.append(min_dist)
            self.sigmas = np.array(sigmas)
        else:
            self.sigmas = None

    def _one_hot_classes(self, y, labels, n_classes) -> np.ndarray:
        """Transforms input labels in one-hot encoding for multi-class problems."""
        Y = np.zeros((y.shape[0], n_classes))
        for c in labels:
            Y[y == c, int(c)] = 1
        return Y

    def _init_model(self, input_size, output_size):
        self.model = LightningModel(
            model=MyLinearLayer(
                input_size=input_size,
                output_size=output_size,
            ),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _fit(self, H, y):
        self._init_model(H.shape[1], len(np.unique(y)))

        custom_train_dataset = CustomDataset(H, y)

        train_loader = DataLoader(dataset=custom_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.trainer = L.Trainer(
            accelerator="auto",
            devices="auto",
            strategy="auto",
            enable_progress_bar=False,
            max_epochs=self.n_epochs,
            precision=self.precision,
        )
        self.trainer.fit(self.model, train_loader)
        self.model.eval()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train GGRBF: compute sigmas and stores structural support vectors."""
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
            self._compute_sigmas(y)
            X_filtered = X
            y_filtered = y
        else:
            # compute membership function (for filtering procedure)
            self.ssv._get_all_neighbors(self.adj)
            quality = self._quality.forward(self.ssv.all_neighbors, y, self.D)
            # filtering procedure
            self._build_perc_thresholds()
            self.valid_samples = self.filter.forward(quality, y, self.percentages)
            valid_index = np.where(np.array(self.valid_samples) == 1)[0]
            # recompute gg
            X_filtered = X[valid_index]
            y_filtered = y[valid_index]
            si = np.where(np.array(self.valid_samples) == 0)[0]
            self._subgg(si)
            # find new ssvs
            self.ssv.fit(self.adj, y_filtered)
            # store ssvs and compute sigmas
            self._compute_sigmas(y_filtered)

        H = self._act_fun.forward(self._dist(X, X_filtered[self.ssv_indexes]), self.sigmas)

        if self.output_weights_algo == "pseudoinverse":
            if self.n_classes > 2:
                y = self._one_hot_classes(y, self.classes_, self.n_classes)
            self.W = np.matmul(np.linalg.pinv(H), y)
        elif self.output_weights_algo == "backpropagation":
            if self.n_classes == 2:
                raise ValueError("backprop only for multi-class problems")
            self._fit(H, y)

        self.ssvs = X_filtered[self.ssv_indexes]
        self.trained = True

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward-pass through the NN."""
        H = self._act_fun.forward(self._dist(X, self.ssvs), self.sigmas)
        if self.output_weights_algo == "pseudoinverse":
            prob = np.matmul(H, self.W)
            if self.n_classes == 2:
                return prob
            elif self.n_classes > 2:
                return prob / prob.sum(axis=1).reshape(-1, 1)
        elif self.output_weights_algo == "backpropagation":
            H = self.model(torch.tensor(H, dtype=torch.float32))
            return self.softmax(H)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns the probability of each class for each sample of the test set."""
        yhat = self.forward(X)
        if self.output_weights_algo == "pseudoinverse":
            if self.n_classes > 2:
                return yhat
            else:
                P_pos = self._sigmoid(yhat)
                P_pos = P_pos.reshape((len(P_pos), 1))
                P_neg = 1 - P_pos
                return np.concatenate((P_neg, P_pos), axis=1)
        elif self.output_weights_algo == "backpropagation":
            return yhat.detach().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns the predicted class for each sample of the test set."""
        yhat = self.forward(X)
        if self.output_weights_algo == "pseudoinverse":
            if self.n_classes > 2:
                return np.argmax(yhat, axis=1)
            else:
                return 1 * (yhat >= 0.5)
        elif self.output_weights_algo == "backpropagation":
            prob = self.forward(X)
            return torch.argmax(prob, axis=1).detach().numpy()
