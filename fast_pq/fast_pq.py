import numpy as np
import sklearn.cluster
from ._transform import transform_data, transform_tables
from ._fast_pq import query_pq_sse, estimate_pq_sse


def pad(arr, mults):
    new_shape = tuple(s + (-s) % m for s, m in zip(arr.shape, mults))
    # TODO: It would be nice to pad using the code with largest possible distance
    # in each dimension, so as to maximiize the likelihood of not seeing any of them
    # in the top values.
    new_arr = np.zeros(new_shape, dtype=arr.dtype)
    new_arr[tuple(slice(0, s) for s in arr.shape)] = arr
    return new_arr


def bottom_k(arr, k):
    """Returns the k smallest indices of arr"""
    if k >= len(arr):
        return np.arange(len(arr))
    return np.argpartition(arr, k)[:k]


class FastPQ:
    def __init__(self, dims_per_block):
        self.dims_per_block = dims_per_block
        self.centers = None  # Shape: (n_blocks, 16, dims_per_block)
        self.center_norms_sq = None  # Shape: (n_blocks, 16)

    def fit(self, data, verbose=False):
        assert data.size > 0, "Can't fit no data"
        data = pad(data, (16, 2 * self.dims_per_block))
        n, d = data.shape
        assert d % self.dims_per_block == 0
        assert (d // self.dims_per_block) % 2 == 0
        dpb = self.dims_per_block
        cl = sklearn.cluster.KMeans(16, n_init=1)
        centers = []
        for i in range(d // self.dims_per_block):
            if verbose:
                print(f"Fitting block {i}")
            cl.fit(data[:, i * dpb : (i + 1) * dpb])
            # It doesn't give too much precision to do separate centers for each block,
            # but durnig queries we need seperate distance tables per block anyway, so
            # it doesn't cost us much.
            centers.append(cl.cluster_centers_.copy())
        self.centers = np.array(centers, dtype=np.float32)
        self.center_norms_sq = np.linalg.norm(centers, axis=2) ** 2
        return self

    def transform(self, data):
        assert self.centers is not None, "PQ has not been fitted"
        if data.size == 0:
            return data
        true_n = data.shape[0]
        data = pad(data, (16, 2 * self.dims_per_block))
        n, d = data.shape
        assert n % 16 == 0
        parts = data.reshape(n, -1, self.dims_per_block)
        ips = np.einsum("ijk,jlk->ijl", parts, self.centers)  # Shape: (n, n_blocks, 16)
        labels = np.argmin(self.center_norms_sq - 2 * ips, axis=2)
        assert labels.shape == (n, d // self.dims_per_block)
        labels = np.array(labels, dtype=np.uint8)
        return true_n, transform_data(labels)

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def distance_table(self, q):
        q = pad(q, (2 * self.dims_per_block,))
        dpb = self.dims_per_block
        n_blocks = q.size / dpb

        parts = q.reshape(-1, dpb)

        # Center the data in the range [-128, 128]
        # TODO: Is this the best scaling formula?
        # TODO: Would this be faster if we used the same centers for each block?
        dists = self.center_norms_sq - 2 * np.einsum("ijk,ik->ij", self.centers, parts)
        # dists += (parts * parts).sum(axis=1, keepdims=True)
        # shift = np.mean(dists)
        # print(np.mean(dists), np.median(dists))
        # shift = 1
        # shift = 128 / n_blocks
        # scale = 128 / (np.max(-(dists-shift)) * np.sqrt(n_blocks))
        # shift = np.mean(dists) / 2
        shift = np.median(dists)
        # shift = 0
        # scale = 1
        scale = 128 / (np.max(np.abs(dists - shift)) * np.sqrt(n_blocks))
        table = np.round(
            (dists - shift) * scale
        )  # Round to nearest integer towards zero.

        # The transformation doesn't care about the sign, so we just use uint
        table = table.astype(np.uint8)
        trans = transform_tables(table)
        return _FastDistanceTable(q, trans, shift, scale, signed=True)

    def udistance_table(self, q):
        # Experimental
        q = pad(q, (2 * self.dims_per_block,))
        dpb = self.dims_per_block
        n_blocks = q.size / dpb
        parts = q.reshape(-1, dpb)
        dists = self.center_norms_sq - 2 * np.einsum("ijk,ik->ij", self.centers, parts)
        shift = np.min(dists)
        dists -= shift
        scale = 255 / (np.max(dists) * np.sqrt(n_blocks))
        table *= scale
        table = table.astype(np.uint8)
        trans = transform_tables(table)
        return _FastDistanceTable(q, trans, shift, scale, signed=False)


class _FastDistanceTable:
    def __init__(self, q, transformed_tables, mean, scale, signed):
        self.q = q
        self.tables = transformed_tables
        self.mean = mean
        self.scale = scale
        self.signed = signed

    def estimate_distances(self, transformed_data, out=None, rescale=False):
        true_n, transformed_data = transformed_data
        if out is None:
            out = np.zeros(2 * len(transformed_data), dtype=np.uint64)
        estimate_pq_sse(transformed_data, self.tables, out, True)
        res = out.view(np.int8)
        res = res[:true_n]  # Trim padding elements
        if not rescale:
            # TODO: Would sorting actually be faster if we casted to float?
            return res
        # The center_norm is already built into res, so we just need to add
        # the norm of q and rescale.
        as_float = np.ascontiguousarray(res, dtype=np.float32)
        return self.q @ self.q + (as_float / self.scale + self.mean)

    def top(self, transformed_data, data, k=1, rescore=None, out=None):
        """
        Find the nearest data points to the query, given the compressed
        and non-compressed data to search.
        """
        if k >= len(data):
            diff = data - self.q
            return np.arange(len(data)), (diff * diff).sum(axis=1)
        if not rescore:
            rescore = min(2 * k + 10, len(data))
        assert rescore >= k

        estimates = self.estimate_distances(transformed_data, out=out, rescale=False)
        guess = bottom_k(estimates, k=rescore)

        diff = data[guess] - self.q
        dists = (diff * diff).sum(axis=1)
        best = bottom_k(dists, k=k)
        return guess[best], dists[best]

    def ctop(self, transformed_data, data, k=1, rescore=None):
        """
        Like top, but uses cython.
        """
        true_n, transformed_data = transformed_data
        k = min(k, true_n)
        if not rescore:
            rescore = min(2 * k + 10, true_n)
        assert true_n >= rescore >= k
        indices = np.zeros((rescore,), dtype=np.int32)
        values = np.zeros((rescore,), dtype=np.int32)
        query_pq_sse(transformed_data, self.tables, indices, values, True)
        good_indices = indices < true_n  # TODO: We remove paddinig in a kinda dumb way
        indices = indices[good_indices]
        if rescore > k:
            diff = data[indices] - self.q[: data.shape[1]]  # Remove padding from q
            dists = np.einsum("ij,ij->i", diff, diff)
            best = bottom_k(dists, k=k)
            return indices[best], dists[best]
        values = values[good_indices]
        return indices, values

    def ctops(self, transformed_datas, data, k=1, rescore=None):
        """
        Like ctop, but does takes multiple datasets at the same time, which might be
        useful when doing multi-probing IVF.
        """
        true_n, transformed_data = transformed_data
        k = min(k, true_n)
        if not rescore:
            rescore = min(2 * k + 10, true_n)
        assert true_n >= rescore >= k

        # Basically, these two arrays define a heap?
        indices = np.zeros((rescore,), dtype=np.int32)
        values = np.zeros((rescore,), dtype=np.int32)
        for true_n, transformed_data in transformed_datas:
            query_pq_sse(transformed_data, self.tables, indices, values, True)

        good_indices = indices < true_n  # TODO: We remove paddinig in a kinda dumb way
        indices = indices[good_indices]
        if rescore > k:
            diff = data[indices] - self.q[: data.shape[1]]  # Remove padding from q
            dists = np.einsum("ij,ij->i", diff, diff)
            best = bottom_k(dists, k=k)
            return indices[best], dists[best]
        values = values[good_indices]
        return indices, values


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
    def __init__(self, q):
        self.q = q

    def estimate_distances(self, data, out=None, rescale=False):
        return ((data - self.q) * (data - self.q)).sum(axis=1)

    def top(self, transformed_data, data, k=1, rescore=None, out=None):
        dists = self.estimate_distances(transformed_data)
        best = bottom_k(dists, k)
        return best, dists[best]
