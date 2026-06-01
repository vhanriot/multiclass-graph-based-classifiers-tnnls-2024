import numpy as np
import torch

from main.models.graph_models.graph.Distance import Distance


class GG:
    def __init__(self) -> None:
        self.dist = Distance(operation="scipy", squared=True, zero_dist_to_inf=True)

    def _dist(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Computes the Euclidian^2 distance matrix between matrices A and B."""
        return self.dist.forward(A, B)

    def classic(self, X: np.ndarray):
        """Computes GG's adjacency matrix in the classic form (2 for loops)

        Args:
            X (np.ndarray): input samples
        Returns:
            F (np.ndarray): distance matrix
            adj (np.ndarray): GG's adjancecy matrix
        """
        n = X.shape[0]
        F = self._dist(X, X)

        adj = np.zeros((n, n), dtype=np.bool_)
        for i in range(n - 1):
            for j in range(i + 1, n):
                min_sum_of_distances = np.min(F[i, :] + F[j, :])
                adj[i, j] = min_sum_of_distances >= F[i, j]
                adj[j, i] = adj[i, j]
        return F, adj

    def classic_in(self, X: np.ndarray):
        """Computes GG's adjacency matrix in the classic form (2 for loops) while also storing how many samples are inside the i,j pair (0 in case it's a GG edge)

        Args:
            X (np.ndarray): input samples
        Returns:
            F (np.ndarray): distance matrix
            adj (np.ndarray): GG's adjancecy matrix
            adj_in (np.ndarray): samples within each i,j hypersphere
        """
        n = X.shape[0]
        F = self._dist(X, X)

        adj = np.zeros((n, n), dtype=np.bool_)
        adj_in = np.zeros((n, n), dtype=np.uint32)
        for i in range(n - 1):
            for j in range(i + 1, n):
                min_sum_of_distances = np.min(F[i, :] + F[j, :])
                adj[i, j] = min_sum_of_distances >= F[i, j]
                adj[j, i] = adj[i, j]
                if not adj[i, j]:
                    samples_inside_sphere = ((F[i, :] + F[j, :]) < F[i, j]).sum()
                    adj_in[i, j] = samples_inside_sphere
                    adj_in[j, i] = samples_inside_sphere
        np.fill_diagonal(adj_in, 1)
        return F, adj, adj_in

    def vectorized(self, X: np.ndarray):
        """Computes GG's adjacency matrix in a vectorized form

        Args:
            X (np.ndarray): input samples
        Returns:
            F (np.ndarray): distance matrix
            adj (np.ndarray): GG's adjancecy matrix
        """
        n = X.shape[0]
        F = self._dist(X, X)

        adj = np.zeros((n, n), dtype=np.bool_)
        for i in range(n - 1):
            A = F[i] + F[i + 1 :]
            idx_min = np.argmin(A, axis=1)
            a = A[np.arange(A.shape[0]), idx_min] - F[i, i + 1 :]
            adj[i, i + 1 :] = np.where(a > 0, 1, 0)
        adj = adj + adj.T
        return F, adj

    def vectorized_in(self, X: np.ndarray):
        """Computes GG's adjacency matrix in a vectorized form while also storing how many samples are inside the i,j pair (0 in case it's a GG edge)

        Args:
            X (np.ndarray): input samples
        Returns:
            F (np.ndarray): distance matrix
            adj (np.ndarray): GG's adjancecy matrix
            adj_in (np.ndarray): samples within each i,j hypersphere
        """
        n = X.shape[0]
        F = self._dist(X, X)

        adj_in = np.zeros((n, n), dtype=np.uint32)
        for i in range(n - 1):
            A = F[i] + F[i + 1 :]
            diff = np.subtract(A, F[i, i + 1 :].reshape(-1, 1))
            samples_inside_sphere_i = np.count_nonzero(diff <= 0, axis=1)
            adj_in[i, i + 1 :] = samples_inside_sphere_i
        adj_in = adj_in + adj_in.T
        np.fill_diagonal(adj_in, 1)
        adj = adj_in == 0
        return F, adj, adj_in

    def sub_GG(self, adj_in, F, si):
        """Computes a second GG from the samples that were not filtered (membership function>threshold) after the first GG computation

        Args:
            adj_in (np.ndarray): samples within each i,j hypersphere
            F (np.ndarray): distance matrix
            si (np.ndarray): samples that were removed from the first GG computation
        Returns:
            adj (np.ndarray): remaining samples' GG's adjancecy matrix
        """
        n = F.shape[0]
        indices = np.setdiff1d(np.arange(n), si)
        F_rm = F[indices][:, si]
        F_min = F[indices][:, indices]
        adj_in_min = adj_in[indices][:, indices]

        n_min = F_rm.shape[0]
        adj = np.zeros((n_min, n_min), dtype=np.bool_)
        for i in range(n_min - 1):
            A = F_rm[i] + F_rm[i + 1 :]
            diff = np.subtract(A, F_min[i, i + 1 :].reshape(-1, 1))
            samples_inside_sphere_i = np.count_nonzero(diff <= 0, axis=1)
            adj[i, i + 1 :] = samples_inside_sphere_i == adj_in_min[i, i + 1 :]
        adj = adj + adj.T
        return adj

    def smallGG(self, F: np.ndarray, m):
        """Computes GG's adjacency matrix in a vectorized form

        Args:
            X (np.ndarray): input samples
        Returns:
            F (np.ndarray): distance matrix
            adj (np.ndarray): GG's adjancecy matrix
        """
        adj = np.zeros((m, m), dtype=np.bool_)
        for i in range(m - 1):
            A = F[i] + F[i + 1 :]
            idx_min = np.argmin(A, axis=1)
            a = A[np.arange(A.shape[0]), idx_min] - F[i + 1 :, i]
            adj[i, i + 1 :] = np.where(a > 0, 1, 0)

        return adj

    def create_fold_partitions(self, n, BATCH_SIZE1, BATCH_SIZE2):
        FOLDS1 = []
        for i in range(0, n, BATCH_SIZE1):
            sub_array = [i, min(i + BATCH_SIZE1, n) - 1]
            FOLDS1.append(sub_array)
        FOLDS2 = []
        for FOLD1 in FOLDS1:
            sub_array = []
            for i in range(FOLD1[0], n, BATCH_SIZE2):
                sub_array.append([i, min(i + BATCH_SIZE2, n) - 1])
            FOLDS2.append(sub_array)
        return FOLDS1, FOLDS2

    def small_batches(self, F_BATCH1, F_BATCH2, FOLD_b2):
        m = F_BATCH1.shape[0]
        adj = np.zeros((m, F_BATCH2.shape[0]), dtype=np.bool_)

        for i in range(m):
            A = F_BATCH1[i] + F_BATCH2
            idx_min = np.argmin(A, axis=1)
            a = A[np.arange(A.shape[0]), idx_min] - F_BATCH1[i, FOLD_b2]
            adj[i] = np.where(a > 0, 1, 0)

        return adj

    def batchGG(self, X, BATCH_SIZE1=1000, BATCH_SIZE2=1000):

        n = X.shape[0]
        BATCH_SIZE1 = np.min([BATCH_SIZE1, n])
        BATCH_SIZE2 = np.min([BATCH_SIZE2, n])

        # creating fold partitions
        FOLDS1, FOLDS2 = self.create_fold_partitions(n, BATCH_SIZE1, BATCH_SIZE2)

        # computing batchGG
        adj = np.zeros((n, n), dtype=np.bool_)
        for i, FOLD1 in enumerate(FOLDS1):
            print(f"FOLD1: {i}")
            F_b1 = self._dist(X[range(FOLD1[0], FOLD1[1] + 1)], X)
            firstB1 = FOLD1[0]
            lastB1 = FOLD1[-1] + 1
            j = 0
            for FOLD2 in FOLDS2[i]:
                j += 1
                print(f"FOLD2: {j}")
                F_b2 = self._dist(X[range(FOLD2[0], FOLD2[1] + 1)], X)
                firstB2 = FOLD2[0]
                lastB2 = FOLD2[-1] + 1
                adj[firstB1:lastB1, firstB2:lastB2] = self.small_batches(
                    F_b1, F_b2, list(range(FOLD2[0], FOLD2[1] + 1))
                )
        adj += adj.T
        return adj

    def small_batches_tensorized(self, F_BATCH1, F_BATCH2, FOLD_b2):
        F_BATCH1 = torch.reshape(F_BATCH1, (F_BATCH1.shape[0], 1, F_BATCH1.shape[1]))
        A = F_BATCH1 + F_BATCH2
        min_dists = torch.min(A, axis=2)[0]
        min_dists = torch.reshape(min_dists, (min_dists.shape[0], 1, min_dists.shape[1]))
        adj = min_dists - F_BATCH1[:, :, FOLD_b2]
        return adj.reshape(adj.shape[0], -1) > 0

    def fully_tensorized(self, X, BATCH_SIZE1=1000, BATCH_SIZE2=1000):

        n = X.shape[0]
        BATCH_SIZE1 = np.min([BATCH_SIZE1, n])
        BATCH_SIZE2 = np.min([BATCH_SIZE2, n])

        # creating fold partitions
        print("creating fold partitions")
        FOLDS1, FOLDS2 = self.create_fold_partitions(n, BATCH_SIZE1, BATCH_SIZE2)

        # computing batchGG (each batch is fully tensorized)
        adj = torch.zeros(n, n, dtype=torch.bool)
        for i, FOLD1 in enumerate(FOLDS1):
            print(f"FOLD1: {i}")
            F_b1 = torch.tensor(self._dist(X[range(FOLD1[0], FOLD1[1] + 1)], X))
            firstB1 = FOLD1[0]
            lastB1 = FOLD1[-1] + 1
            j = 0
            for FOLD2 in FOLDS2[i]:
                j += 1
                print(f"FOLD2: {j}")
                F_b2 = torch.tensor(self._dist(X[range(FOLD2[0], FOLD2[1] + 1)], X))
                firstB2 = FOLD2[0]
                lastB2 = FOLD2[-1] + 1
                adj[firstB1:lastB1, firstB2:lastB2] = self.small_batches_tensorized(
                    F_b1, F_b2, list(range(FOLD2[0], FOLD2[1] + 1))
                )
        adj = adj.numpy()
        adj += adj.T
        return adj

    def small_batches_tensorized_in(self, F_BATCH1, F_BATCH2, FOLD_b2):
        F_BATCH1 = torch.reshape(F_BATCH1, (F_BATCH1.shape[0], 1, F_BATCH1.shape[1]))
        A = F_BATCH1 + F_BATCH2
        diag_ij = F_BATCH1[:, :, FOLD_b2]
        diag_ij = torch.reshape(diag_ij, (diag_ij.shape[0], diag_ij.shape[2], 1))
        diff = A - diag_ij
        adj_in = torch.sum(diff < 0, axis=2)
        return adj_in == 0, adj_in

    def fully_tensorized_in(self, X, BATCH_SIZE1=1000, BATCH_SIZE2=1000):

        n = X.shape[0]
        BATCH_SIZE1 = np.min([BATCH_SIZE1, n])
        BATCH_SIZE2 = np.min([BATCH_SIZE2, n])

        # creating fold partitions
        FOLDS1, FOLDS2 = self.create_fold_partitions(n, BATCH_SIZE1, BATCH_SIZE2)

        # computing batchGG (each batch is fully tensorized)
        adj = torch.zeros(n, n, dtype=torch.bool)
        adj_in = torch.zeros(n, n, dtype=torch.int32)
        for i, FOLD1 in enumerate(FOLDS1):
            F_b1 = torch.tensor(self._dist(X[range(FOLD1[0], FOLD1[1] + 1)], X))
            firstB1 = FOLD1[0]
            lastB1 = FOLD1[-1] + 1
            for FOLD2 in FOLDS2[i]:
                F_b2 = torch.tensor(self._dist(X[range(FOLD2[0], FOLD2[1] + 1)], X))
                firstB2 = FOLD2[0]
                lastB2 = FOLD2[-1] + 1
                adj[firstB1:lastB1, firstB2:lastB2], adj_in[firstB1:lastB1, firstB2:lastB2] = (
                    self.small_batches_tensorized_in(F_b1, F_b2, list(range(FOLD2[0], FOLD2[1] + 1)))
                )
        adj = adj.numpy()
        adj_in = adj_in.numpy()
        return adj, adj_in
