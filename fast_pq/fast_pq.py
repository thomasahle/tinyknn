import numpy as np
import sklearn.cluster
from collections import namedtuple

import warnings
from sklearn.exceptions import ConvergenceWarning

from ._transform import transform_data, transform_tables
from ._fast_pq import query_pq_sse, estimate_pq_sse, init_heap

from numpy.core._methods import _amax, _mean

warnings.simplefilter("error", category=ConvergenceWarning)

TransformedData = namedtuple("TransformedData", "size packed")

import os

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"


def pad1(arr, m):
    (s,) = arr.shape
    new_shape = (s + (-s) % m,)
    padded_arr = np.zeros(new_shape, dtype=arr.dtype)
    padded_arr[:s] = arr
    return padded_arr


def pad2(arr, m1, m2):
    s1, s2 = arr.shape
    new_shape = (s1 + (-s1) % m1, s2 + (-s2) % m2)
    padded_arr = np.zeros(new_shape, dtype=arr.dtype)
    padded_arr[:s1, :s2] = arr
    return padded_arr


def bottom_k(arr, k):
    if k >= len(arr):
        return np.arange(len(arr))
    return np.argpartition(arr, k)[:k]


def bottom_k_2d(arr, k):
    if k >= arr.shape[1]:
        return np.resize(np.arange(arr.shape[1]), arr.shape)
    return np.argpartition(arr, k, axis=1)[:, :k]


class FastPQ:
    def __init__(self, dims_per_block):
        """
        Initializes the FastPQ class with the specified number of dimensions per block.

        Parameters
        ----------
        dims_per_block : int
            The number of dimensions per block.
        """
        self.dims_per_block = dims_per_block
        self.centers = None  # Shape: (n_blocks, 16, dims_per_block)
        self.sqrt_n_blocks = None

    def fit(self, data, verbose=False):
        """
        Fits the FastPQ model to the given data.

        Parameters
        ----------
        data : array-like
            The input data to fit the FastPQ model.
        verbose : bool, optional, default=False
            If True, prints additional information during the fitting process.

        Returns
        -------
        self : FastPQ
            The fitted FastPQ model.
        """
        _transformed_data = self.fit_transform(data, verbose)
        return self

    def fit_transform(self, data, verbose=False):
        assert data.size > 0, "Can't fit no data"
        true_n = data.shape[0]
        # SSE assumes the number of rows is divisible by 16.
        # It also needs the number of columns to be even, so we pad to a multiple
        # of 2 * self.dims_per_block.
        data = pad2(data, 16, 2 * self.dims_per_block)
        n, d = data.shape
        assert d % self.dims_per_block == 0
        assert (d // self.dims_per_block) % 2 == 0
        dpb = self.dims_per_block
        # We always use 16 clusters in FastPQ, since we want to use 4 bit SSE operations.
        cl = sklearn.cluster.KMeans(16, n_init=2)
        centers = []
        transformed = []
        for i in range(d // self.dims_per_block):
            if verbose:
                print(f"Fitting block {i}")
            try:
                cl.fit(data[:, i * dpb : (i + 1) * dpb])
            except ConvergenceWarning:
                pass
            # It doesn't give too much precision to do separate centers for each block,
            # but durnig queries we need seperate distance tables per block anyway, so
            # it doesn't cost us much.
            # We .copy() the centers because we are reusing the KMeans object.
            centers.append(cl.cluster_centers_.copy())
            # Might as well grab the labels (transformed column) while we have it.
            transformed.append(cl.labels_[:, None])
        # (d / dpb, 16, dpb) -> (16, d)
        self.centers = (
            np.array(centers, dtype=np.float32).transpose(1, 0, 2).reshape(16, d)
        )
        self.sqrt_n_blocks = np.sqrt(d // dpb)

        labels = np.hstack(transformed).astype(np.uint8)
        return TransformedData(true_n, transform_data(labels))

    def transform(self, data):
        """
        Transforms the given data using the FastPQ model.

        Parameters
        ----------
        data : array-like
            The input data to transform using the FastPQ model.

        Returns
        -------
        tuple
            A tuple containing the transformed data and the number of true elements in the data.
        """
        assert self.centers is not None, "PQ has not been fitted"
        if data.size == 0:
            return data
        true_n = data.shape[0]
        data = pad2(data, 16, 2 * self.dims_per_block)
        n, d = data.shape
        assert n % 16 == 0
        dpb = self.dims_per_block
        blocks = d // dpb
        dists = (
            np.square(self.centers[None] - data[:, None])
            .reshape(n, 16, blocks, dpb)
            .sum(axis=-1)
        )
        labels = np.argmin(dists, axis=1).astype(np.uint8)
        assert labels.shape == (n, blocks)
        return TransformedData(true_n, transform_data(labels))

    def distance_table(self, q):
        """
        Computes the distance table for the given query vector q.

        Parameters
        ----------
        q : array-like
            The query vector.

        Returns
        -------
        _FastDistanceTable
            The FastDistanceTable object containing the computed distance table.
        """
        dpb = self.dims_per_block
        q = pad1(q, 2 * dpb)

        diff = (self.centers - q).reshape(16, -1, dpb)
        dists = np.einsum("ijk,ijk->ij", diff, diff)

        # We do this by shifting by the mean value and scaling by the nuber of blocks.
        # The idea is that we don't want to have an overflow as we add together
        # distances in uint8 format. The median seems to work better here than the mean,
        # even though it's a bit more expensive to compute. It turns out the squared distances
        # are roughly exponentially distributed, so the median is ~ mean * log(2).
        shift = _mean(dists) * 0.6931471806
        dists -= shift
        scale = 128 / (_amax(dists) * self.sqrt_n_blocks)
        table = np.round(dists * scale)

        # The transformation doesn't care about the sign, so we just use uint
        table = table.astype(np.uint8)
        trans = transform_tables(table.T)
        return _FastDistanceTable(q, trans, shift, scale, signed=True)

    def udistance_table(self, q):
        """
        Computes the unsigned distance table for the given query vector q.
        Currently experimental.

        Parameters
        ----------
        q : array-like
            The query vector.

        Returns
        -------
        _FastDistanceTable
            The FastDistanceTable object containing the computed unsigned distance table.
        """
        dpb = self.dims_per_block
        q = pad1(q, 2 * dpb)
        n_blocks = q.size // dpb
        dists = np.square(self.centers - q).reshape(16, n_blocks, dpb).sum(axis=-1)
        shift = np.min(dists)
        dists -= shift
        scale = 255 / (np.max(dists) * np.log(n_blocks) * np.sqrt(n_blocks))
        table = np.round(dists * scale)
        table = table.astype(np.uint8)
        trans = transform_tables(table.T)
        return _FastDistanceTable(q, trans, shift, scale, signed=False)


class _FastDistanceTable:
    def __init__(self, q, transformed_tables, mean, scale, signed):
        self.q = q
        self.tables = transformed_tables
        self.mean = mean
        self.scale = scale
        self.signed = signed

    def __repr__(self):
        return (
            f"FastDistanceTable(q={self.q}, tables={self.tables}, mean={self.mean}, "
            "scale={self.scale}, signed={self.signed})"
        )

    def estimate_distances(self, transformed_data, out=None, rescale=False):
        true_n, transformed_data = transformed_data
        if out is None:
            out = np.zeros(2 * len(transformed_data), dtype=np.uint64)
        estimate_pq_sse(transformed_data, self.tables, out, self.signed)
        res = out.view(np.int8)
        res = res[:true_n]  # Trim padding elements
        if not rescale:
            return res
        # The center_norm is already built into res, so we just need to add
        # the norm of q and rescale.
        as_float = np.ascontiguousarray(res, dtype=np.float32)
        return self.q @ self.q + (as_float / self.scale + self.mean)

    def top(self, transformed_data, data, k=1, rescore=None):
        """
        Find the nearest data points to the query, given the compressed
        and non-compressed data to search.
        """
        true_n, transformed_data = transformed_data
        assert len(data) == true_n
        k = min(k, true_n)
        # In the first pass we collect `rescore` many rows
        if not rescore:
            rescore = min(2 * k + 10, true_n)
        assert true_n >= rescore >= k

        indices = np.zeros((rescore,), dtype=np.int64)
        values = np.zeros((rescore,), dtype=np.int32)
        init_heap(indices, values, self.signed)
        query_pq_sse(
            transformed_data, true_n, self.tables, indices, values, self.signed
        )

        # In a second pass we compute the true distances and return the actually
        # closest points. If we got fewer or exactly k outputs, there is no need
        # to compute the true distaneces.
        if rescore <= k:
            return indices, values

        # Remove padding from q
        diff = data[indices] - self.q[: data.shape[1]]
        dists = np.einsum("ij,ij->i", diff, diff)
        best = bottom_k(dists, k=k)
        return indices[best], dists[best]


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
