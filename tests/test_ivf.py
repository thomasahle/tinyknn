import numpy as np
import pytest

from fast_pq import FastPQ, DummyPQ
from fast_pq import IVF, cdist, brute


def test_cdist():
    n1, n2, d = 10, 8, 5
    np.random.seed(10)
    for chunk in [1, 10, 100]:
        X = np.random.randn(n1, d)
        Y = np.random.randn(n2, d)
        dists = cdist(X, Y, chunk=chunk)
        for i in range(n1):
            for j in range(n2):
                tru_dist = np.sum((X[i] - Y[j]) ** 2)
                assert np.isclose(dists[i, j], tru_dist)


def test_brute():
    n1, n2, d = 40, 28, 5
    np.random.seed(10)
    X = np.random.randn(n1, d)
    Y = np.random.randn(n2, d)
    expected = cdist(X, Y).argpartition(axis=1, kth=10)[:, :10]
    best = brute(X, Y, 10)
    assert np.all(np.sort(expected) == np.sort(best))


def test_euclidian_recall():
    np.random.seed(10)
    assert _test_recall_inner(10**2, 20, 10, 2, 10, "euclidean", 1) > 0.1
    assert _test_recall_inner(10**2, 20, 10, 2, 10, "euclidean", 2) > 0.2
    assert _test_recall_inner(10**2, 20, 10, 2, 10, "euclidean", 4) > 0.35
    assert _test_recall_inner(10**2, 20, 10, 2, 10, "euclidean", 8) > 0.50


def test_angular_recall():
    np.random.seed(10)
    assert _test_recall_inner(10**2, 20, 10, 2, 10, "angular", 1) > 0.09
    assert _test_recall_inner(10**2, 20, 10, 2, 10, "angular", 2) > 0.18
    assert _test_recall_inner(10**2, 20, 10, 2, 10, "angular", 4) > 0.27
    assert _test_recall_inner(10**2, 20, 10, 2, 10, "angular", 8) > 0.36


def _test_recall_inner(n, d, nq, dpb, at, metric, n_probes):
    X = np.random.randn(n, d).astype(np.float32)
    qs = np.random.randn(nq, d).astype(np.float32)
    if at < n:
        trus = cdist(qs, X).argpartition(axis=1, kth=at)[:, :at]
    else:
        trus = np.broadcast_to(np.arange(n), (nq, n))
    pq = FastPQ(dims_per_block=dpb)
    ivf = IVF(metric, int(n**0.5), pq)
    ivf.fit(X).build(X)
    recall_at = 0
    for q, tru in zip(qs, trus):
        guess = ivf.query(q, k=at, n_probes=n_probes)
        recall_at += len(set(guess) & set(tru))
    return recall_at / nq / at


@pytest.mark.filterwarnings("ignore:Number of distinct clusters")
def test_small():
    np.random.seed(10)
    assert _test_recall_inner(15, 10, 30, 2, 10, "euclidean", 1) > 0.05
