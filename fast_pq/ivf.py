import numpy as np
import sklearn.cluster
from fast_pq import bottom_k, bottom_k_2d
from ._fast_pq import query_pq_sse, init_heap
import time
from contextlib import contextmanager


@contextmanager
def timer(verbose, text):
    if verbose:
        print(text)
        start = time.time()
    yield
    if verbose:
        print(f"Took {time.time() - start:.1f}s")


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
            part = -2 * Xn[i : i + chunk] @ Yn.T
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
        assert pq.centers is not None, "PQ should be pre-fitted"
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
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                cl.fit(X)
                self.all_centers = cl.cluster_centers_
                self.all_centers /= np.linalg.norm(self.all_centers, axis=1, keepdims=True)

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

        assert n_probes <= self.n_clusters
        self.data = data = X.copy()

        with timer(verbose, "Computing nearest clusters..."):
            # TODO: Use compression here, somehow.

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

        t0, t1, t2 = 0, 0, 0
        with timer(verbose, "Transforming points and adding them to their nearest centers..."):
            for i in range(self.active_centers.shape[0]):
                # TODO: This is a lot slower than it needs to be.
                s = time.time()
                mask = np.any(nearest_indices == i, axis=1)
                t0 += time.time() - s

                # TODO: We should be able to extract the relevant data
                # directly from a previously transformed X.
                # However, that would require slicing into the compressed QuickADC format,
                # which is annoying.
                s = time.time()
                self.pq_transformed_points[i] = self.pq.transform(
                    np.ascontiguousarray(data[mask])
                )
                t1 += time.time() - s

                s = time.time()
                self.ids[i] = np.arange(data.shape[0], dtype=np.int64)[mask]
                t2 += time.time() - s

        if verbose:
            print(f"Finding any: {t0:.1f}")
            print(f"Transforming: {t1:.1f}")
            print(f"Computing ids: {t2:.1f}")

        return self

    def query(self, q, k, n_probes=1):
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
        top, _ = dtable.top(
            self.pq_transformed_centers, self.active_centers, k=n_probes
        )

        # For the first pass, get 2k candidates from each cluster
        # One may experiment with tuning this. Could even make it an argument to query.
        rescore = (n_probes + 1) * k + 1
        indices = np.empty((rescore,), dtype=np.int64)
        # We need a scratch space to store the approximate values,
        # but we won't actually use it.
        values = np.empty((rescore,), dtype=np.int32)
        indices[:] = -1
        values[:] = 127
        #init_heap(indices, values, True)

        for i, cl in enumerate(top):
            true_n, transformed_data = self.pq_transformed_points[cl]
            query_pq_sse(
                transformed_data, true_n, dtable.tables, indices, values, True, labels=self.ids[cl]
            )

        # Remove duplicates (only an issue if build_probes > 1)
        indices = np.unique(indices)

        # We may have gotten some heap padding elements mixed in
        if indices.size != 0 and indices[0] == -1:
            indices = indices[1:]

        # If we have less or equal to k values, there's no point in rescoring
        if len(indices) <= k:
            return indices

        q = q[: self.data.shape[1]]  # Remove padding from q
        diff = self.data[indices] - q
        dists = np.einsum("ij,ij->i", diff, diff)
        best = bottom_k(dists, k=k)
        return indices[best]
