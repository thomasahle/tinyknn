import numpy as np
import sklearn.cluster
import _transform as transform
from _fast_pq import query_pq_sse


class PQ:
    def __init__(self, dims_per_block):
        self.dims_per_block = dims_per_block
        self.centers = []

    def fit_transform(self, data):
        n, d = data.shape
        assert d % self.dims_per_block == 0
        assert (d // self.dims_per_block) % 2 == 0
        assert n % 16 == 0
        dpb = self.dims_per_block
        blocks = d // self.dims_per_block
        cl = sklearn.cluster.KMeans(16, n_init=1)
        centers, labels = [], []
        for i in range(blocks):
            cl.fit(data[:, i * dpb : (i + 1) * dpb])
            centers.append(cl.cluster_centers_)
            labels.append(cl.labels_)
        self.centers = np.array(centers, dtype=np.float32)
        self.norms = np.linalg.norm(centers, axis=2) ** 2
        print(self.centers.shape)
        labels = np.array(labels, dtype=np.uint8).T
        return transform.transform_data(labels)

    def transform_query(self, q):
        dpb = self.dims_per_block
        parts = q.reshape(-1, dpb)

        # TODO: This can probably be speed up by a better memory layout
        ips = np.einsum("ijk,ik->ij", self.centers, parts)
        norms = np.linalg.norm(parts, axis=1, keepdims=True) ** 2
        table = self.norms - 2 * ips + norms

        # TODO: Try if median or something would work in place of max here
        blocks = q.size / dpb
        # scale = np.mean(table) / 255 * blocks
        scale = np.max(table) / 255 * np.sqrt(blocks)
        table = np.floor(table / scale).astype(np.uint8)

        trans = transform.transform_tables(table)
        return trans, scale

    def distances(self, data, tables):
        out = np.zeros(2 * len(data), dtype=np.uint64)
        query_pq_sse(data, tables, out)
        return out.view(np.uint8)
