import numpy as np
import sklearn.cluster
from collections import namedtuple
from scipy.stats import ortho_group
import tqdm

import warnings
from sklearn.exceptions import ConvergenceWarning

from ._transform import transform_data, transform_tables

from ._fast_pq import init_heap
from .utils import knn_brute, knn_brute1, pad2, pad1

from numpy.core._methods import _amax, _mean

warnings.simplefilter("error", category=ConvergenceWarning)



avx = True
if avx:
    from ._fast_pq_avx import query_pq_avx as query_pq, estimate_pq_avx as estimate_pq
    dpad = 4
else:
    from ._fast_pq import query_pq_sse as query_pq, estimate_pq_sse as estimate_pq
    dpad = 2


TransformedData = namedtuple("TransformedData", "size packed")


class FastPQ:
    def __init__(self, dims_per_block, use_kmeans=True, rotate_dim=64):
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
        self.use_kmeans = use_kmeans
        self.rotate_dim = rotate_dim
        self.R = None

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
        assert data.size > 0, "Can't fit no data"
        true_n, true_d = data.shape

        # SSE assumes the number of rows is divisible by 16.
        # It also needs the number of columns to be even, so we pad to a multiple
        # of 2 * self.dims_per_block.
        # We always use 16 clusters in FastPQ, since we want to use 4 bit SSE operations.
        data = pad2(data, 16, dpad * self.dims_per_block)
        n, d = data.shape
        dpb = self.dims_per_block

        if self.rotate_dim is not None and true_d != 100:
            self.R = ortho_group.rvs(dim=d)
            if d > self.rotate_dim:
                d = self.rotate_dim
                self.R = self.R[:d]
            data = data @ self.R.T

        # if self.use_opq:
        #     assert False, "Not implemented"
        #     for _ in range(10):
        #         centers = self._fit_code(data, verbose=verbose)
        #         cols = data.reshape(n, d // dpb, dpb).transpose(1, 0, 2)
        #         labels = [knn_brute(col, code, 1) for col, code in zip(cols, centers)]
        #     # Stack the codebooks along a new axis
        #     centers_stacked = np.stack(centers, axis=1)  # Shape: (K, M, dpb)
        #     # Convert the labels list into a 2D array
        #     labels_array = np.column_stack(labels)  # Shape: (n, M)
        #     # Select the corresponding centers for each label
        #     selected_centers = centers_stacked[labels_array]  # Shape: (n, M, dpb)
        centers = self._fit_code(data, verbose=verbose)

        # (d / dpb, 16, dpb) -> (16, d)
        self.centers = (
            np.array(centers, dtype=np.float32).transpose(1, 0, 2).reshape(16, d)
        )
        self.sqrt_n_blocks = np.sqrt(d // dpb)

        return self

    def fit_transform(self, data, verbose=False):
        return self.fit(data, verbose).transform(data, verbose)

    def _fit_code(self, data, verbose=False):
        """ Returns the best PQ code (d / dpb, 16, dpb) """
        n, d = data.shape
        dpb = self.dims_per_block
        cols = data.reshape(n, d // dpb, dpb).transpose(1, 0, 2)

        centers = []
        if self.use_kmeans:
            cl = sklearn.cluster.KMeans(16, n_init=2)
            loop = range(d // dpb)
            for i in loop if not verbose else tqdm.tqdm(loop):
                try:
                    cl.fit(cols[i])
                except ConvergenceWarning:
                    pass
                # We .copy() the centers because we are reusing the KMeans object.
                centers.append(cl.cluster_centers_.copy())
        else:
            assert dpb == 2, "Fixed code only defined for dpb = 2"
            # Standard code for quantizing a Gaussian
            base = np.array(
                [(0, 0)]
                + [
                    (r * np.cos(th), r * np.sin(th))
                    for r, num_points in zip([1, 2], [6, 9])
                    for th in np.linspace(0, 2 * np.pi, num_points, endpoint=False)
                ]
            )
            # Scale separately for each column
            for i in range(d // self.dims_per_block):
                col = cols[i]
                # Transform base code
                mu = np.mean(col, axis=0)
                S = np.cov(col.T, bias=True)
                code = base @ np.linalg.cholesky(S).T + mu
                centers.append(code)
        return centers

    def transform(self, data, verbose=False):
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
        data = pad2(data, 16, dpad * self.dims_per_block)
        n, d = data.shape

        if self.R is not None:
            data = data @ self.R.T
            d = data.shape[1]

        dpb = self.dims_per_block
        blocks = d // dpb

        cols = data.reshape(n, d // dpb, dpb).transpose(1, 0, 2)
        code_cols = self.centers.reshape(16, -1, dpb).transpose(1, 0, 2)
        labels = []
        pairs = zip(cols, code_cols)
        if verbose:
            pairs = tqdm.tqdm(pairs, total=len(cols))
        labels = [knn_brute(col, code, 1) for col, code in pairs]
        labels = np.hstack(labels).astype(np.uint8)
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
        raw_q = q
        q = pad1(q, dpad * dpb)
        if self.R is not None:
            q = q @ self.R.T

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
        return _FastDistanceTable(q, raw_q, trans, shift, scale, signed=True)

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
        raw_q = q
        q = pad1(q, dpad * dpb)
        if self.R is not None:
            q = q @ self.R.T
        n_blocks = q.size // dpb
        dists = np.square(self.centers - q).reshape(16, n_blocks, dpb).sum(axis=-1)
        shift = np.min(dists)
        dists -= shift
        scale = 255 / (np.max(dists) * np.log(n_blocks) * np.sqrt(n_blocks))
        table = np.round(dists * scale)
        table = table.astype(np.uint8)
        trans = transform_tables(table.T)
        return _FastDistanceTable(q, raw_q, trans, shift, scale, signed=False)


class _FastDistanceTable:
    def __init__(self, q, raw_q, transformed_tables, mean, scale, signed):
        self.q = q
        self.raw_q = raw_q
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
        estimate_pq(transformed_data, self.tables, out, self.signed)
        res = out.view(np.int8 if self.signed else np.uint8)
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
        query_pq(
            transformed_data, true_n, self.tables, indices, values, self.signed
        )

        # In a second pass we compute the true distances and return the actually
        # closest points. If we got fewer or exactly k outputs, there is no need
        # to compute the true distaneces.
        if rescore <= k:
            return indices

        # Use raw (unpadded, unrotated) q when comparing with raw data
        best = knn_brute1(self.raw_q, data[indices], k)
        return indices[best]
