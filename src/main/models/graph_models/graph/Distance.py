import numpy as np
from scipy.spatial import distance


class Distance:
    def __init__(self, operation="scipy", squared=False, zero_dist_to_inf=False) -> None:
        self.operation = operation
        self.squared = squared
        self.zero_dist_to_inf = zero_dist_to_inf

    def _loop(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Computes the Euclidian distance matrix between matrices A and B in a loop

        Args:
            A (np.ndarray): matrix1 (biggest matrix)
            B (np.ndarray): matrix2
        """
        m1 = A.shape[0]
        m2 = B.shape[0]
        D = np.zeros((m1, m2))
        for i in range(m2):
            D[:, i] = np.sum((A - B[i]) ** 2, axis=1)
        return np.sqrt(D)

    def _tensor(self, A: np.ndarray, B: np.ndarray):
        """Computes the Euclidian distance matrix between matrices A and B using numpy tensors

        Args:
            A (np.ndarray): matrix1 (biggest matrix)
            B (np.ndarray): matrix2
        """
        A = A.reshape((A.shape[0], 1, A.shape[1]))
        B = B.reshape((1, B.shape[0], B.shape[1]))
        Z = A * A
        Z = Z + B * B
        Z = Z - 2 * A * B
        Z = Z.sum(axis=2)
        D = np.sqrt(Z)
        return D

    def _scipy(self, A: np.ndarray, B: np.ndarray):
        """Computes the Euclidian distance matrix between matrices A and B using scipy

        Args:
            A (np.ndarray): matrix1 (biggest matrix)
            B (np.ndarray): matrix2
        """
        return distance.cdist(A, B, "euclidean")

    def forward(self, A, B):
        if self.operation == "loop":
            D = self._loop(A, B)
        elif self.operation == "scipy":
            D = self._scipy(A, B)
        elif self.operation == "tensor":
            D = self._tensor(A, B)
        else:
            raise ValueError("Distance operation not implemented")

        if self.squared:
            D = D**2

        if self.zero_dist_to_inf:
            D[D == 0] = np.inf

        return D
