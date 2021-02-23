import numpy as np
import scipy as sp

from _fast_pq import estimate_pq_sse, query_pq_sse
from fast_pq import FastPQ


def test_recall():
    np.random.seed(10)
    for i in range(1, 10):
        for method in ['argpartition', 'top', 'ctop']:
            recall_at_10 = _test_recall_inner(16 * i, 8 * i, 100, 2, method)
            assert recall_at_10 > 0.8


def _test_recall_inner(n, d, k, dpb, method):
    X = np.random.randn(n, d).astype(np.float32)
    qs = np.random.randn(k, d).astype(np.float32)
    trus = sp.spatial.distance.cdist(qs, X).argmin(axis=1)

    pq = FastPQ(dims_per_block=dpb)
    data = pq.fit_transform(X)
    recall_at_10 = 0
    for q, tru in zip(qs, trus):
        dtable = pq.distance_table(q)
        if method == 'argpartition':
            top10 = dtable.estimate_distances(data).argpartition(10)[:10]
        elif method == 'top':
            top10, _ = dtable.top(data, X, 10)
        elif method == 'ctop':
            top10, _ = dtable.ctop(data, X, 10)
        if tru in top10:
            recall_at_10 += 1
    return recall_at_10 / k


def test_topk():
    np.random.seed(11)
    for i in range(2, 4):
        for signed in [True, False]:
            _test_topk_inner(16 * i, 3, 100, 2, signed)


def _test_topk_inner(n, k, d, dpb, signed):
    X = np.random.randn(n, d).astype(np.float32)
    qs = np.random.randn(k, d).astype(np.float32)
    pq = FastPQ(dims_per_block=dpb)
    _, data = pq.fit_transform(X)

    out = np.zeros(2 * len(data), dtype=np.uint64)
    indices = np.zeros((n,), dtype=np.int32)
    values = np.zeros((n,), dtype=np.int32)

    for q in qs:
        dtable = pq.distance_table(q)

        estimate_pq_sse(data, dtable.tables, out, signed)
        est = out.view(np.int8 if signed else np.uint8)

        query_pq_sse(data, dtable.tables, indices, values, signed)

        # print(est.argsort(kind='stable'))
        # print(sorted(est))
        # print(indices)
        # print(values)

        est.sort()
        assert np.all(est == values)
