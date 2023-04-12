import numpy as np
import sklearn.cluster
from fast_pq import bottom_k
from ._fast_pq import query_pq_sse


def cdist(X, Y, chunk=100):
    """
    Computes the squared Euclidean distances between two sets of points X and Y.
    Returns R st. R[i,j] = dist(X_i, Y_j)^2.
    Equivalent to scipy.spatial.distance.cdist.
    """
    # This is how sklearn computes row norms. It 
    # %timeit np.linalg.norm(x, axis=1)
    # 486 ms ± 6.33 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # %timeit np.einsum('ij,ij->i',x,x)
    # 62.9 ms ± 2.35 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    # %timeit (x*x).sum(axis=1)
    # 495 ms ± 6.71 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    nx = np.einsum("ij,ij->i", X, X)
    ny = np.einsum("ij,ij->i", Y, Y)
    res = np.zeros((nx.size, ny.size))
    for i in range(0, nx.size, chunk):
        res[i : i + chunk] = nx[i : i + chunk, None] + ny
        res[i : i + chunk] -= 2 * X[i : i + chunk] @ Y.T
    return res


def brute(X, Y, k, metric="euclidean", chunk=100):
    """
    Computes the k-nearest neighbors for each point in X based on the squared Euclidean distances
    between X and Y.
    """
    nx = np.einsum("ij,ij->i", X, X)
    ny = np.einsum("ij,ij->i", Y, Y)
    res = np.zeros((nx.size, k))
    if metric == "euclidean":
        for i in range(0, nx.size, chunk):
            part = nx[i : i + chunk, None] + ny - 2 * X[i : i + chunk] @ Y.T
            res[i : i + chunk] = part.argpartition(axis=1, kth=k)[:, :k]
    elif metric == "angular":
        Xn = X / np.sqrt(nx[:, None])
        Yn = Y / np.sqrt(ny[:, None])
        for i in range(0, nx.size, chunk):
            part = - 2 * Xn[i : i + chunk] @ Yn.T
            res[i : i + chunk] = part.argpartition(axis=1, kth=k)[:, :k]
    else:
        raise ValueError(f"Metric not supported: {metric}")
    return res


def brute1(x, Y, k):
    """
    Computes the k-nearest neighbors for each point in X based on the squared Euclidean distances
    between X and Y.
    """
    diff = Y - x
    dists = np.einsum("ij,ij->i", diff, diff)
    return dists.argpartition(kth=k)[:k]


class IVF:
    def __init__(self, metric, n_clusters, pq):
        assert metric in ["euclidean", "angular"]
        self.metric = metric
        self.pq = pq
        self.pq_transformed_points = [None] * n_clusters
        self.pq_transformed_centers = [None] * n_clusters
        self.n_clusters = n_clusters
        self.ids = [None] * n_clusters

    def fit(self, X, verbose=False):
        """
        Decides on IVF centers and fits product quantizer.
        Doesn't insert any points init the data structure.
        If you want faster training, just fit on a sample rather than all the data
        like this: ivf.fit(X[np.random.choice(X.shape[0], 10**4, replace=False)]
        """

        n, d = X.shape
        X = np.ascontiguousarray(X, dtype=np.float32)
        cl = sklearn.cluster.KMeans(
            n_clusters=self.n_clusters, n_init=1, verbose=verbose
        )

        if verbose:
            print("Fitting IVF cluster centers")

        if self.metric == "euclidean":
            cl.fit(X)
            self.all_centers = cl.cluster_centers_

        elif self.metric == "angular":
            # For angular we use kmeans clustering, but normalize the norm of the centers
            # as to make inner product equivalent to angular similarity.
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            cl.fit(X)
            self.all_centers = cl.cluster_centers_
            self.all_centers /= np.linalg.norm(self.all_centers, axis=1, keepdims=True)

        # We use a single product quantizer for everything
        if verbose:
            print("Fitting PQ")
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

        self.data = data = X.copy()

        if verbose:
            print("Computing nearest clusters...")

        if self.metric == "euclidean":
            distances = cdist(data, self.all_centers)
        elif self.metric == "angular":
            data /= np.linalg.norm(data, axis=1, keepdims=True)
            distances = - data @ self.all_centers.T

        nearest_indices = np.argpartition(distances, n_probes, axis=1)[:, :n_probes]

        # We make sure that all centers have at least one point.
        # This shouldn't be a problem unless `n` is nearly as small as
        # `n_clusters`.

        self.active_centers = np.ascontiguousarray(
            self.all_centers[np.unique(nearest_indices.flatten())], dtype=np.float32
        )
        self.pq_transformed_centers = self.pq.transform(self.active_centers)

        for i in range(self.active_centers.shape[0]):
            mask = np.any(nearest_indices == i, axis=1)
            self.pq_transformed_points[i] = self.pq.transform(np.ascontiguousarray(data[mask]))
            true_n = self.pq_transformed_points[i][0]
            self.ids[i] = np.arange(data.shape[0])[mask]

        return self

    def query(self, q, k, n_probes=1, rescore_centers=None, rescore_lists=None):
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
        rescore_centers : array-like, optional, default=None
            The transformed center data used for rescoring. If None, the original centers are used.
        rescore_lists : array-like, optional, default=None
            The transformed list data used for rescoring. If None, the original lists are used.

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
        top, _ = dtable.ctop(self.pq_transformed_centers, self.active_centers, k=n_probes)

        # For the first pass, get 2k candidates from each cluster
        # One may experiment with tuning this. Could even try decreasing it
        # for further away cluster centers like "late move reductions" in Stockfish.
        rescore = 2*k + 1
        indices = np.empty((n_probes, rescore), dtype=np.int32)
        # We need a scratch space to store the approximate values,
        # but we won't actually use it.
        values = np.empty((rescore,), dtype=np.int32)

        for i, cl in enumerate(top):
            true_n, transformed_data = self.pq_transformed_points[cl]
            query_pq_sse(transformed_data, true_n, dtable.tables,
                         indices[i], values, True)
            # Convert to global indices
            indices[i] = self.ids[cl][indices[i]]

        # Remove duplicates (only an issue if build_probes > 1)
        indices = np.unique(indices.flatten())

        # If we have less or equal to k values, there's no point in rescoring
        if len(indices) <= k:
            return indices

        q = q[: self.data.shape[1]] # Remove padding from q
        diff = self.data[indices] - q
        dists = np.einsum("ij,ij->i", diff, diff)
        best = bottom_k(dists, k=k)
        return indices[best]

