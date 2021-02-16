import numpy as np
import sklearn.cluster

class IVF:
    def __init__(self, metric, n_clusters, pq):
        assert metric in ['euclidean', 'angular']
        self.metric = metric
        self.pq = pq
        self.pq_transformed_points = [None] * n_clusters
        self.pq_transformed_centers = [None] * n_clusters
        self.n_clusters = n_clusters
        self.lists, self.ids = [None]*n_clusters, [None]*n_clusters
        self.norms = [None]*n_clusters # ||x||^2 - Only used for euclidean metric

    def fit(self, X, verbose=False):
        # If you want faster training, just fit on a sample rather than all the data
        # like this: ivf.fit(X[np.random.choice(X.shape[0], 10**4, replace=False)]

        n, d = X.shape
        X = np.ascontiguousarray(X, dtype=np.float32)
        cl = sklearn.cluster.KMeans(n_clusters = self.n_clusters, n_init=1)

        if verbose:
            print('Fitting IVF cluster centers')

        if self.metric == 'euclidean':
            cl.fit(X)
            centers = cl.cluster_centers_
            labels = cl.labels_

        elif self.metric == 'angular':
            # For angular we use kmeans clustering, but normalize the norm of the centers
            # as to make inner product equivalent to angular similarity.
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            cl.fit(X)
            centers = cl.cluster_centers_
            centers /= np.linalg.norm(centers, axis=1, keepdims=True)
            labels = np.argmax(X @ centers.T, axis=1)

        # We make sure that all centers have at least one point.
        # This shouldn't be a problem unless `n` is nearly as small as
        # `n_clusters`.
        used_labels = np.unique(labels)
        self.centers = np.ascontiguousarray(centers[used_labels], dtype=np.float32)
        self.n_clusters = self.centers.shape[0]

        if verbose:
            print('Fitting PQ')

        # We use a single product quantizer for everything
        self.pq.fit(X)
        self.pq_transformed_centers = self.pq.transform(self.centers)

        # Move data around to localize individual clusters
        for i in range(self.n_clusters):
            mask = labels == i
            # TODO: QuckADC stores the resiuals (X[mask] - center) here.
            # Is that better? That would require seperate PQs for each cluster...
            self.lists[i] = np.ascontiguousarray(X[mask])
            self.pq_transformed_points[i] = self.pq.transform(self.lists[i])
            self.ids[i] = np.array(range(n))[mask]
            self.norms[i] = np.linalg.norm(self.lists[i], axis=1) ** 2

    def query(self, q, k, n_probes=1, rescore_centers=None, rescore_lists=None):
        q = np.ascontiguousarray(q, dtype=np.float32)
        if self.metric == 'angular':
            q /= np.linalg.norm(q)
        dtable = self.pq.distance_table(q)

        # Shortcut for single-probe
        if n_probes == 1:
            i = dtable.top(self.pq_transformed_centers, self.centers)[0][0]
            js, _ = dtable.top(self.pq_transformed_points[i], self.lists[i], k=k)
            return self.ids[i][js]

        # Find best centers
        top, _ = dtable.top(self.pq_transformed_centers, self.centers, k=n_probes)
        # TODO: Preallocate space for js and dists rather than dynamically like this.
        # (Even better: Keep that space allocated between queries)
        js, dists = [], []
        for i in top:
            sub_n, _ = self.lists[i].shape
            sub_js, sub_dists = dtable.top(self.pq_transformed_points[i], self.lists[i], k=k)
            # Translate into global indexing
            js.append(self.ids[i][sub_js])
            dists.append(sub_dists)

        # Merge datas
        js, dists = np.concatenate(js), np.concatenate(dists)
        best = np.argpartition(dists, kth=k)[:k]
        return js[best]

