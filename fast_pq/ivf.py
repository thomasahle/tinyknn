import numpy as np
import sklearn.cluster
from fast_pq import FastPQ
from .utils import bottom_k, bottom_k_2d, cdist, timer, brute1
from ._fast_pq import query_pq_sse, init_heap


def group_data_by_indices(X, indices, k):
    """
    Given a 2D array `X` of shape (N, d), a 2D array `indices` of shape (N, c) with integers in [0, k),
    and an integer `k`, return a list `parts` of k arrays, such that all rows X[i] are in parts[indices[i, j]] for some j.

    Args:
    X (np.ndarray): A 2D numpy array of shape (N, d) containing the data points.
    indices (np.ndarray): A 2D numpy array of shape (N, c) containing integers in the range [0, k).
    k (int): The number of groups.

    Returns:
    list: A list of k numpy arrays, where each array contains the rows of X that belong to the corresponding group.
    """
    # Initialize an empty list to store the data points for each group
    assert 0 <= np.min(indices) and np.max(indices) < k
    parts = [[] for _ in range(k)]
    ids = [[] for _ in range(k)]

    # Iterate over each column in indices
    for i in range(indices.shape[1]):
        # Get the current column of indices
        col = indices[:, i]

        # Sort X by the current column of indices
        sorted_indices = np.argsort(col)
        sorted_X = X[sorted_indices]

        # Compute the length of the runs in the current column of indices (sorted)
        sorted_indices_unique, run_lengths = np.unique(
            col[sorted_indices], return_counts=True
        )
        # Use a loop only over j = 0 to k - 1 that selects the runs of X and adds them to the corresponding group in parts
        start = 0
        for i, run_length in enumerate(run_lengths):
            end = start + run_length
            parts[sorted_indices_unique[i]].append(sorted_X[start:end])
            ids[sorted_indices_unique[i]].append(sorted_indices[start:end])
            start = end

    parts = [np.vstack(part) for part in parts]
    ids = [np.hstack(id_list) for id_list in ids]
    return parts, ids


class IVF:
    def __init__(self, metric, n_clusters, pq=None):
        assert metric in ["euclidean", "angular"]
        self.metric = metric
        self.pq = FastPQ(dims_per_block=2) if pq is None else pq
        assert self.pq.centers is None, "PQ should not be pre-fitted"
        self.pq_transformed_points = [None] * n_clusters
        self.pq_transformed_centers = [None] * n_clusters
        self.n_clusters = n_clusters
        self.ids = [None] * n_clusters

    def fit(self, X, verbose=False):
        """
        Decides on IVF centers using full X, not PQ transformed.
        Doesn't insert any points into the data structure.
        If you want faster training, just fit on a sample rather than all the data
        like this: ivf.fit(X[np.random.choice(X.shape[0], 10**4, replace=False)]
        """

        n, d = X.shape
        assert n >= 1

        with timer(verbose, "Fitting IVF cluster centers..."):
            X = np.ascontiguousarray(X, dtype=np.float32)
            cl = sklearn.cluster.KMeans(
                n_clusters=self.n_clusters, n_init=1, verbose=verbose
            )

            if self.metric == "euclidean":
                cl.fit(X)
                self.all_centers = cl.cluster_centers_

            elif self.metric == "angular":
                # For angular we use kmeans clustering, but normalize the norm of the centers
                # as to make inner product equivalent to angular similarity.
                X = X / np.linalg.norm(X, axis=1, keepdims=True)
                cl.fit(X)
                self.all_centers = cl.cluster_centers_
                self.all_centers /= np.linalg.norm(
                    self.all_centers, axis=1, keepdims=True
                )

        with timer(verbose, "Fitting PQ to data..."):
            self.pq.fit(X, verbose=verbose)

        return self

    def build(self, X, n_probes=2, verbose=False):
        """
        Builds the data structure by associating each data point in X with its nearest n_probes cluster centers.

        Parameters:
        -----------
        X : numpy.ndarray, shape (n_samples, n_features)
            The input data points to be indexed. Each row represents a data point with n_features dimensions.
        n_probes : int, optional, default: 2
            The number of nearest cluster centers to assign each data point to. Must be a positive integer.
        verbose : bool, optional, default: False
            If True, print additional information during the index building process.

        Returns:
        --------
        self : object
            Returns the instance itself, with the index structure built and data points assigned to the nearest n_probes cluster centers.
        """

        assert (
            n_probes <= self.n_clusters
        ), f"Can't assign points to {n_probes} clusters, as index only has {self.n_clusters}"
        self.data = data = X.copy()

        with timer(verbose, "Computing nearest clusters..."):
            # TODO: Use the compression here, somehow.

            if self.metric == "euclidean":
                distances = cdist(data, self.all_centers)
            elif self.metric == "angular":
                data /= np.linalg.norm(data, axis=1, keepdims=True)
                distances = -data @ self.all_centers.T

            nearest_indices = bottom_k_2d(distances, n_probes)

        # We make sure that all centers have at least one point. This is just
        # a simple optimiation that's useful when there's not a lot of data
        # in the IVF, or if the query is OOD.
        with timer(verbose, "PQ Transforming active centers..."):
            self.active_centers = np.ascontiguousarray(
                self.all_centers[np.unique(nearest_indices)], dtype=np.float32
            )
            self.pq_transformed_centers = self.pq.transform(self.active_centers)

        with timer(verbose, "Transforming points..."):
            groups, self.ids = group_data_by_indices(
                data, nearest_indices, self.active_centers.shape[0]
            )
            for i in range(self.active_centers.shape[0]):
                self.pq_transformed_points[i] = self.pq.transform(groups[i])

        return self

    def query(self, q, k, n_probes=1, pass_1=None):
        """
        Queries the data structure to find the top k closest elements to the given query vector q.

        Parameters
        ----------
        q : array-like
            The query vector for which to find the top k closest elements.
        k : int
            The number of closest elements to return.
        n_probes : int, optional, default=1
            The number of probes to use for searching the closest elements.

        Returns
        -------
        numpy.ndarray
            An array of indices corresponding to the top k closest elements to the query vector q.
        """

        q = np.ascontiguousarray(q, dtype=np.float32)
        if self.metric == "angular":
            q /= np.linalg.norm(q)
        dtable = self.pq.distance_table(q)

        # Find best centers
        top = dtable.top(self.pq_transformed_centers, self.active_centers, k=n_probes)

        # For the first pass, get 2k candidates from each cluster
        # One may experiment with tuning this. Could even make it an argument to query.
        if pass_1 is None:
            pass_1 = (n_probes + 1) * k + 1
        indices = np.full(pass_1, -1, dtype=np.int64)
        values = np.full(pass_1, 127, dtype=np.int32)

        for i, cl in enumerate(top):
            true_n, transformed_data = self.pq_transformed_points[cl]
            query_pq_sse(
                transformed_data,
                true_n,
                dtable.tables,
                indices,
                values,
                True,
                labels=self.ids[cl],
            )

        # We may have gotten some heap padding elements mixed in. This is very rarely an issue,
        # so we check before we spend time building a mask and indexing.
        if -1 in indices:
            indices = indices[indices != -1]

        # If we have less or equal to k values, there's no point in rescoring
        if len(indices) <= k:
            return indices

        unpadded_q = q[: self.data.shape[1]]
        best = brute1(unpadded_q, self.data[indices], k)
        return indices[best]
