import numpy as np
import sklearn.cluster
from _transform import transform_data, transform_tables
from _fast_pq import query_pq_sse


class FastPQ:
    def __init__(self, dims_per_block):
        self.dims_per_block = dims_per_block
        self.centers = []      # Shape: (n_blocks, 16, dims_per_block)
        self.center_norms = [] # Shape: (n_blocks, 16)

    def fit(self, data):
        n, d = data.shape
        dpb = self.dims_per_block
        assert d % self.dims_per_block == 0
        assert (d // self.dims_per_block) % 2 == 0
        assert n % 16 == 0
        cl = sklearn.cluster.KMeans(16, n_init=1)
        centers = []
        for i in range(d // self.dims_per_block):
            cl.fit(data[:, i * dpb : (i + 1) * dpb])
            centers.append(cl.cluster_centers_)
        self.centers = np.array(centers, dtype=np.float32)
        self.center_norms = np.linalg.norm(centers, axis=2) ** 2
        return self

    def transform(self, data):
        n, d = data.shape
        parts = data.reshape(n, -1, self.dims_per_block)
        ips = np.einsum("ijk,jlk->ijl", parts, self.centers) # Shape: (n, n_blocks, 16)
        labels = np.argmin(self.center_norms - 2*ips, axis=2)
        assert labels.shape == (n, d//self.dims_per_block)
        labels = np.array(labels, dtype=np.uint8)
        return transform_data(labels)

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def distance_table(self, q, signed=False):
        dpb = self.dims_per_block
        parts = q.reshape(-1, dpb)
        blocks = q.size / dpb

        # TODO: This can probably be speed up by a better memory layout
        ips = np.einsum("ijk,ik->ij", self.centers, parts)
        if signed:
            # TODO: Tune this formula
            scale = 128 / np.max(ips) / np.sqrt(blocks)
            table = np.round(ips * scale)
        else:
            norms = (parts*parts).sum(axis=1, keepdims=True)
            #norms = np.linalg.norm(parts, axis=1, keepdims=True) ** 2
            table = self.center_norms - 2 * ips + norms
            # TODO: Try if median or something would work in place of max here
            # scale = np.mean(table) / 255 * blocks
            scale = 255 / np.max(table) / np.sqrt(blocks)
            table = np.floor(table * scale)

        # The transformation doesn't care about the sign, so we just use uint
        # In either case.
        table = table.astype(np.uint8)
        trans = transform_tables(table)
        return _FastDistanceTable(q, trans, scale, signed)

class _FastDistanceTable:
    def __init__(self, q, transformed_tables, scale, signed):
        self.q = q
        self.tables = transformed_tables
        self.scale = scale
        self.signed = signed

    def estimate_distances(self, transformed_data, out=None, rescale=False):
        if out is None:
            out = np.zeros(2 * len(transformed_data), dtype=np.uint64)
        query_pq_sse(transformed_data, self.tables, out, self.signed)
        res = out.view(np.int8) if self.signed else out.view(np.uint8)
        if rescale:
            return np.ascontiguousarray(res, dtype=np.float32) * self.scale
        return res

    def top(self, transformed_data, data, k=1, rescore=None, out=None):
        '''
        Returns (indxs, dists) or (indxs, -ips) if signed.
        Indices are in no particular order.
        '''
        if not rescore:
            rescore = min(2*k+10, len(data))
        assert rescore >= k
        est = self.estimate_distances(transformed_data, out=out)
        est = est.astype(np.float32) # Numpy only works on floats
        if self.signed:
            guess = np.argpartition(-est, rescore)[:rescore] # We want big ips
            ips = q @ data[guess].T
            best = np.argpartition(-ips, k)[:k]
            return guess[best], -ips[best]
        else:
            guess = np.argpartition(est, rescore)[:rescore] # We want small dists
            dists = np.sum(data[guess] * data[guess], axis=1, keepdims=True) - 2 * q @ data[guess].T
            best = np.argpartition(dists, k)[:k]
            return guess[best], dists[best]

class DummyPQ:
    def fit(self, data):
        return self

    def transform(self, data):
        return data

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def distance_table(self, q):
        return DummyDistanceTable(q)

class DummyDistanceTable:
    def __init__(self, q, signed):
        self.q = q
        self.signed = signed

    def estimate_distances(self, data, out=None, rescale=False):
        if self.signed:
            return q @ data.T
        return (data * data).sum(axis=1, keepdims=True) - 2 * q @ data.T

    def top(self, transformed_data, data, k=1, rescore=None, out=None):
        dists = self.estimate_distances(transformed_data)
        if self.signed:
            dists = -dists
        best = np.argpartition(dists, k)[:k]
        return best, dists[best]

