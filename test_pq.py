import pytest
import numpy as np
import scipy as sp

from fast_pq import PQ


def test_recall():
    for i in range(1, 10):
        recall_at_10 = _test_recall_inner(16 * i, 8 * i, 100, 2)
        assert recall_at_10 > 0.8


def _test_recall_inner(n, d, k, dpb):
    X = np.random.randn(n, d).astype(np.float32)
    qs = np.random.randn(k, d).astype(np.float32)
    trus = sp.spatial.distance.cdist(qs, X).argmin(axis=1)
    pq = PQ(dims_per_block=dpb)
    data = pq.fit_transform(X)
    recall_at_10 = 0
    for q, tru in zip(qs, trus):
        tables, _ = pq.transform_query(q)
        top10 = pq.distances(data, tables).argpartition(10)[:10]
        if tru in top10:
            recall_at_10 += 1
    return recall_at_10 / k
