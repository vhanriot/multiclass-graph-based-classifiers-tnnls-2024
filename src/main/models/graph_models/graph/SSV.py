import numpy as np


class SSV:
    def __init__(self) -> None:
        pass

    def _get_all_neighbors(self, adj) -> None:
        """Generates a list where each element is a numpy array with all the graph-neighbors (based on the adjacency matrix) of the ith sample.
        Ex: all_neighbors[0] has a numpy array with all the GG-neighbors of sample 0
            all_neighbors[100] has a numpy array with all the GG-neighbors of sample 100
        """
        pairs = np.transpose(np.array(np.where(adj == True)))
        all_neighbors = [[] for _ in range(len(adj))]
        for i in range(len(pairs)):
            all_neighbors[pairs[i, 0]].append(pairs[i, 1])
        all_neighbors = [np.unique(all_neighbors[i]) for i in range(len(all_neighbors))]
        self.all_neighbors = all_neighbors

    def _build_pairs(self, adj) -> None:
        """Generates an array where each element corresponds to the indexes of a GG-pair."""
        self._get_all_neighbors(adj)

        original_samples = []
        neighbors = []
        for i in range(len(adj)):
            n_neighbors = len(self.all_neighbors[i])
            original_samples.extend(np.repeat(i, n_neighbors))
            neighbors.extend(self.all_neighbors[i])
        all_pairs = np.c_[original_samples, neighbors]
        all_pairs = np.sort(all_pairs, axis=1)
        all_pairs = np.unique(all_pairs, axis=0)
        self.all_pairs = all_pairs

    def _build_ssv_pairs(self, y: np.ndarray) -> None:
        """Generates an array where each element corresponds to the indexes of a SSV-pair (which is a GG-pair with different classes)."""
        bound_pairs = [
            self.all_pairs[i].tolist()
            for i in range(len(self.all_pairs))
            if y[self.all_pairs[i][0]] != y[self.all_pairs[i][1]]
        ]
        self.bound_pairs = np.array(bound_pairs)

    def fit(self, adj, y) -> None:
        """Defines pairs and SSVs pairs."""
        assert adj.shape[0] == y.shape[0], (
            f"adj and y should have the same number of samples. adj.shape={adj.shape}, y.shape={y.shape}"
        )
        self._build_pairs(adj)
        self._build_ssv_pairs(y)
