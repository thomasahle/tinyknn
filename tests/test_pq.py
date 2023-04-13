import numpy as np
import pytest

from fast_pq._fast_pq import estimate_pq_sse, query_pq_sse, init_heap
from fast_pq import FastPQ, cdist


def test_recall():
    np.random.seed(10)
    for i in range(1, 5):
        for method in ["argpartition", "top"]:
            n = np.random.randint(16 * i, 16 * (i + 1))
            recall_at_10 = _test_recall_inner(n, 8 * i, 100, 2, method)
            assert recall_at_10 > 0.8


def _test_recall_inner(n, d, k, dpb, method):
    X = np.random.randn(n, d).astype(np.float32)
    qs = np.random.randn(k, d).astype(np.float32)
    trus = cdist(qs, X).argmin(axis=1)

    pq = FastPQ(dims_per_block=dpb)
    data = pq.fit_transform(X)
    recall_at_10 = 0
    for q, tru in zip(qs, trus):
        dtable = pq.distance_table(q)
        if method == "argpartition":
            top10 = dtable.estimate_distances(data).argpartition(10)[:10]
        elif method == "top":
            top10, _ = dtable.top(data, X, 10)
        if tru in top10:
            recall_at_10 += 1
    return recall_at_10 / k


@pytest.mark.filterwarnings("ignore:Number of distinct clusters")
def test_topk():
    np.random.seed(11)
    for n in tuple(range(1, 10)) + (20, 30, 50):
        for dpb in [1, 2]:
            for signed in [True, False]:
                _test_topk_inner(n, 3, 11, dpb, signed)


@pytest.mark.filterwarnings("ignore:Number of distinct clusters")
def test_topk_0():
    np.random.seed(11)
    for signed in [True, False]:
        with pytest.raises(AssertionError):
            _test_topk_inner(0, 3, 11, 2, signed)


def test_fit_transform():
    np.random.seed(11)
    n, d = 100, 10
    X = np.random.randn(n, d).astype(np.float32)

    pq = FastPQ(2)
    n0, tdata0 = pq.fit_transform(X)
    n1, tdata1 = pq.transform(X)
    assert n0 == n1
    np.testing.assert_array_equal(tdata0, tdata1)


def _test_topk_inner(n, m, d, dpb, signed):
    X = np.random.randn(n, d).astype(np.float32)
    qs = np.random.randn(m, d).astype(np.float32)
    pq = FastPQ(dims_per_block=dpb)
    _, data = pq.fit_transform(X)

    for q in qs:
        dtable = pq.distance_table(q)

        out = np.zeros(2 * len(data), dtype=np.uint64)
        estimate_pq_sse(data, dtable.tables, out, signed)
        est = out.view(np.int8 if signed else np.uint8)
        est = est[:n] # Remove padding

        k = n
        indices = np.zeros((k,), dtype=np.int64)
        values = np.zeros((k,), dtype=np.int32)
        init_heap(indices, values, signed)
        query_pq_sse(data, n, dtable.tables, indices, values, signed)

        # Remove padding, and ignore
        # mask = indices < n |
        maxv = 127 if signed else 255
        mask = values < maxv  # we don't guarantee returing things with the max value
        indices, values = indices[mask], values[mask]
        values.sort()

        est.sort()
        est = est[est < maxv]
        assert np.all(est == values)


def test_large_labels():
    n, d, k = 100, 10, 100
    X = np.random.randn(n, d).astype(np.float32)
    q = np.random.randn(d).astype(np.float32)

    pq = FastPQ(2)
    _, data = pq.fit_transform(X)
    dtable = pq.distance_table(q)

    indices = np.empty((k,), dtype=np.int64)
    values = np.empty((k,), dtype=np.int32)
    labels = np.arange(n, dtype=np.int64) + 10**12
    init_heap(indices, values, True)
    query_pq_sse(data, n, dtable.tables, indices, values, True, labels)
    indices.sort()
    np.testing.assert_array_equal(indices, labels)

