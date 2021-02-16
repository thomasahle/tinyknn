import numpy as np
import scipy as sp

from fast_pq import FastPQ, DummyPQ
from ivf import IVF



def test_euclidian_recall():
    np.random.seed(10)
    assert _test_recall_inner(10**3, 100, 10**2, 2, 10, 'euclidean', 1) > .1
    assert _test_recall_inner(10**3, 100, 10**2, 2, 10, 'euclidean', 2) > .2
    assert _test_recall_inner(10**3, 100, 10**2, 2, 10, 'euclidean', 4) > .35
    assert _test_recall_inner(10**3, 100, 10**2, 2, 10, 'euclidean', 8) > .55

def test_angular_recall():
    np.random.seed(10)
    assert _test_recall_inner(10**3, 100, 10**2, 2, 10, 'angular', 1) > .09
    assert _test_recall_inner(10**3, 100, 10**2, 2, 10, 'angular', 2) > .18
    assert _test_recall_inner(10**3, 100, 10**2, 2, 10, 'angular', 4) > .27
    assert _test_recall_inner(10**3, 100, 10**2, 2, 10, 'angular', 8) > .36


def _test_recall_inner(n, d, nq, dpb, at, metric, n_probes):
    X = np.random.randn(n, d).astype(np.float32)
    qs = np.random.randn(nq, d).astype(np.float32)
    trus = sp.spatial.distance.cdist(qs, X).argpartition(axis=1, kth=at)[:,:at]
    pq = FastPQ(dims_per_block=dpb)
    ivf = IVF(metric, int(n**.5), pq)
    ivf.fit(X)
    recall_at = 0
    for q, tru in zip(qs, trus):
        guess = ivf.query(q, k=at, n_probes=n_probes)
        recall_at += len(set(guess) & set(tru))
    return recall_at / nq / at
