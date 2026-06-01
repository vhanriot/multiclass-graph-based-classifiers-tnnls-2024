import warnings
from typing import Optional

import numpy as np


class Quality:
    def __init__(self, quality: str, sigma: Optional[float]) -> None:
        if quality == "cardinality" and sigma is not None:
            warnings.warn("Chipclass membership function was used with not none sigma")
        if quality == "dist_based" and sigma is None:
            raise ValueError("Distance based quality index needs a sigma value.")
        self.quality = quality
        self.sigma = sigma

    def Cardinality(self, all_neighbors: list, y: np.ndarray) -> np.ndarray:
        """Computes the membership function considering the cardinality of a set

        Args:
        all_neighbors: list of np.ndarray graph-neighbors for each sample i
        y: labels
        """
        quality = [np.sum(y[all_neighbors[i]] == yi) / len(all_neighbors[i]) for i, yi in enumerate(y)]
        return np.array(quality)

    def DistanceBased(self, all_neighbors: list, y: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Computes the membership function considering the distance between the sample and its graph-neighbors

        Args:
        all_neighbors: list of np.ndarray graph-neighbors for each sample i
        D: squared distance matrix of the input set
        y: training set labels
        """
        if self.sigma is None:
            raise ValueError("DistanceBased should'nt be called if sigma is null.")
        quality = []
        for i, yi in enumerate(y):
            squared_dist_sample_to_neighbors = D[i][all_neighbors[i]]
            assert (
                np.any(np.isinf(squared_dist_sample_to_neighbors)) == False
            )  # make sure no inf dist exists between i and its neighbors (it shouldnt happen unless duplicate samples in the training data)
            kernel = np.exp(-squared_dist_sample_to_neighbors / (2 * self.sigma**2 + 1e-32))
            quality.append(np.sum(kernel[y[all_neighbors[i]] == yi]) / np.sum(kernel) + 1e-32)
        return np.array(quality)

    def forward(self, all_neighbors: list, y: np.ndarray, D: np.ndarray | None = None) -> np.ndarray:
        """Applying the activation function."""
        if self.quality == "cardinality":
            return self.Cardinality(all_neighbors, y)
        elif self.quality == "dist_based":
            return self.DistanceBased(all_neighbors, y, D)
        else:
            raise ValueError("This filter doesn't exist: try 'cardinality' or 'dist_based'")
