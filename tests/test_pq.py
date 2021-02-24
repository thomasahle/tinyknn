import numpy as np
import scipy as sp
import pytest

from _fast_pq import estimate_pq_sse, query_pq_sse
from fast_pq import FastPQ


def test_recall():
    np.random.seed(10)
    for i in range(1, 5):
        for method in ['argpartition', 'top', 'ctop']:
            n = np.random.randint(16*i, 16*(i+1))
            recall_at_10 = _test_recall_inner(n, 8 * i, 100, 2, method)
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


@pytest.mark.filterwarnings("ignore:Number of distinct clusters")
def test_topk():
    np.random.seed(11)
    for n in tuple(range(1, 10)) + (20, 30, 50):
        for signed in [True, False]:
            _test_topk_inner(n, 3, 11, 2, signed)

@pytest.mark.filterwarnings("ignore:Number of distinct clusters")
def test_topk_0():
    np.random.seed(11)
    for signed in [True, False]:
        with pytest.raises(AssertionError):
            _test_topk_inner(0, 3, 11, 2, signed)

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
        #est = est[:n] # Remove padding

        k = n + (-n)%16 # Round up to nearest 16 because of padding
        indices = np.zeros((k,), dtype=np.int32)
        values = np.zeros((k,), dtype=np.int32)
        query_pq_sse(data, dtable.tables, indices, values, signed)

        print('esti', est.argsort(kind='stable'))
        print('topi', indices)
        print('estv', sorted(est))
        print('topv', values)
        print()

        # Remove padding, and ignore 
        #mask = indices < n |  
        maxv = 127 if signed else 255
        mask = values < maxv # we don't guarantee returing things with the max value
        indices, values = indices[mask], values[mask] 

        est.sort()
        est = est[est < maxv]
        assert np.all(est == values)
