import numpy as np
import sklearn.cluster


def cdist(X, Y, chunk=100):
    """ Returns R st. R[i,j] = dist(X_i, Y_j)^2 """
    nx = np.einsum("ij,ij->i", X, X)  # This is how sklearn computes rownorms
    ny = np.einsum("ij,ij->i", Y, Y)
    res = np.zeros((nx.size, ny.size))
    for i in range(0, nx.size, chunk):
        res[i : i + chunk] = np.add.outer(nx[i : i + chunk], ny)
        res[i : i + chunk] -= 2 * X[i : i + chunk] @ Y.T
    return res


def brute(X, Y, k, chunk=100):
    """ Returns R st. R[i,j] = dist(X_i, Y_j)^2 """
    nx = np.einsum("ij,ij->i", X, X)  # This is how sklearn computes rownorms
    ny = np.einsum("ij,ij->i", Y, Y)
    res = np.zeros((nx.size, k))
    for i in range(0, nx.size, chunk):
        part = np.add.outer(nx[i : i + chunk], ny) - 2 * X[i : i + chunk] @ Y.T
        res[i : i + chunk] = part.argpartition(axis=1, kth=k)[:, :k]
    return res


class IVF:
    def __init__(self, metric, n_clusters, pq):
        assert metric in ["euclidean", "angular"]
        self.metric = metric
        self.pq = pq
        self.pq_transformed_points = [None] * n_clusters
        self.pq_transformed_centers = [None] * n_clusters
        self.n_clusters = n_clusters
        self.lists, self.ids = [None] * n_clusters, [None] * n_clusters

    def fit(self, X, verbose=False):
        # Decides on IVF centers and fits product quantizer.
        # Doesn't insert any points init the data structure.
        # If you want faster training, just fit on a sample rather than all the data
        # like this: ivf.fit(X[np.random.choice(X.shape[0], 10**4, replace=False)]

        n, d = X.shape
        X = np.ascontiguousarray(X, dtype=np.float32)
        cl = sklearn.cluster.KMeans(n_clusters=self.n_clusters, n_init=1)

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
        self.pq.fit(X)

        return self

    def build(self, X, verbose=False):

        if self.metric == "euclidean":
            labels = cdist(X, self.all_centers).argmin(axis=1)
        elif self.metric == "angular":
            labels = np.argmax(X @ self.all_centers.T, axis=1)

        # We make sure that all centers have at least one point.
        # This shouldn't be a problem unless `n` is nearly as small as
        # `n_clusters`.
        used_labels = np.unique(labels)
        self.active_centers = np.ascontiguousarray(
            self.all_centers[used_labels], dtype=np.float32
        )
        self.pq_transformed_centers = self.pq.transform(self.active_centers)

        # Move data around to localize individual clusters
        for i in range(used_labels.size):
            mask = labels == i
            # TODO: QuckADC stores the resiuals (X[mask] - center) here.
            # Is that better? That would require seperate PQs for each cluster...
            self.lists[i] = np.ascontiguousarray(X[mask])
            self.pq_transformed_points[i] = self.pq.transform(self.lists[i])
            self.ids[i] = np.arange(X.shape[0])[mask]

        return self

    def query(self, q, k, n_probes=1, rescore_centers=None, rescore_lists=None):
        q = np.ascontiguousarray(q, dtype=np.float32)
        if self.metric == "angular":
            q /= np.linalg.norm(q)
        dtable = self.pq.distance_table(q)

        # Shortcut for single-probe
        # if n_probes == 1:
        #    i = dtable.top(self.pq_transformed_centers, self.centers)[0][0]
        #    js, _ = dtable.top(self.pq_transformed_points[i], self.lists[i], k=k)
        #    return self.ids[i][js]

        # Find best centers
        top, _ = dtable.top(
            self.pq_transformed_centers, self.active_centers, k=n_probes
        )
        # TODO: Preallocate space for js and dists rather than dynamically like this.
        # (Even better: Keep that space allocated between queries)
        js, dists = [], []
        for i in top:
            sub_n, _ = self.lists[i].shape
            sub_js, sub_dists = dtable.top(
                self.pq_transformed_points[i], self.lists[i], k=k
            )
            # Translate into global indexing
            js.append(self.ids[i][sub_js])
            dists.append(sub_dists)

        # Merge datas
        js, dists = np.concatenate(js), np.concatenate(dists)
        if k >= len(js):
            return js
        best = np.argpartition(dists, kth=k)[:k]
        return js[best]
