import numpy as np
import sklearn.cluster
from fast_pq import bottom_k


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


def brute1(x, Y, k):
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
        self.lists, self.ids = [None] * n_clusters, [None] * n_clusters

    def fit(self, X, verbose=False):
        # Decides on IVF centers and fits product quantizer.
        # Doesn't insert any points init the data structure.
        # If you want faster training, just fit on a sample rather than all the data
        # like this: ivf.fit(X[np.random.choice(X.shape[0], 10**4, replace=False)]

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
            # That is, even if we used the same centers we would still have to
            # compute a seperate distance table for each partition we visit.
            self.lists[i] = np.ascontiguousarray(X[mask])
            self.pq_transformed_points[i] = self.pq.transform(self.lists[i])
            self.ids[i] = np.arange(X.shape[0])[mask]

        self.data = X

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
        top, _ = dtable.ctop(
            self.pq_transformed_centers, self.active_centers, k=n_probes
        )
        # TODO: Preallocate space for js and dists rather than dynamically like this.
        # (Even better: Keep that space allocated between queries)
        js, dists = [], []
        for i in top:
            sub_n, _ = self.lists[i].shape
            sub_js, sub_dists = dtable.ctop(
                self.pq_transformed_points[i], self.lists[i], k=k
            )
            # Translate into global indexing
            js.append(self.ids[i][sub_js])
            dists.append(sub_dists)

        # Merge datas
        if not js:
            return js  # Concatenate doesn't work on empty lists
        js, dists = np.concatenate(js), np.concatenate(dists)
        if k >= len(js):
            return js
        best = np.argpartition(dists, kth=k)[:k]
        return js[best]

    # Ide til ny query:
    # First find centers, using plenty of rescoring. Maybe even just precise.
    # Then run query_sse on each cluster with a sufficiently large k.
    # Let t be the size of the largest cluster.
    # After each query_sse replace indices < t with idmap[ids]+t.
    # Don't change the value list.
    # Once done, subtract t from id list to get a list of the real ids.
    # This is then used for rescoring.
    #
    # If I start having a performance problem due to the update time of the
    # insertion sort, maybe I should start using a real priority queue.
    # I guess I can implement that quickly enough.

    def query3(self, q, k, n_probes=1):
        dtable = pq.distance_table(q)

    def query2(self, q, k, n_probes=1, rescore_centers=None, rescore_lists=None):
        q = np.ascontiguousarray(q, dtype=np.float32)
        if self.metric == "angular":
            q /= np.linalg.norm(q)
        dtable = self.pq.distance_table(q)

        # Find best centers
        # top = brute1(q, self.active_centers, n_probes)
        # top = bottom_k(dtable.estimate_distances(self.pq_transformed_centers), n_probes)
        c_dists = dtable.estimate_distances(self.pq_transformed_centers)
        top = bottom_k(c_dists, 2 * n_probes + 10)
        # print(c_dists[top])
        top = top[brute1(q, self.active_centers[top], k=n_probes)]

        # TODO: Preallocate space for js and dists rather than dynamically like this.
        # (Even better: Keep that space allocated between queries)
        js, dists = [], []
        for i in top:
            sub_dists = dtable.estimate_distances(self.pq_transformed_points[i])
            js.append(self.ids[i])
            dists.append(sub_dists)
        js, dists = np.concatenate(js), np.concatenate(dists)

        # Merge datas
        if k >= len(js):
            return js
        rescore = min(2 * k + 10, len(js))
        best_ids_1 = bottom_k(dists, rescore)
        # print(dists[best_ids_1])
        # print()
        diffs = self.data[js[best_ids_1]] - q
        real_dists = (diffs * diffs).sum(axis=1)
        best = bottom_k(real_dists, k)
        return js[best_ids_1[best]]
