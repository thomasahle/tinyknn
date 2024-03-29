import pytest
import numpy as np
from tinyknn import IVF, FastPQ, knn_brute

np.random.seed(10)

n = 1000
d = 10
nq = 30
at = 10
dpb = 2

X = np.random.randn(n, d).astype(np.float32)
qs = np.random.randn(nq, d).astype(np.float32)


def compute_recall(metric, build_probes, query_probes):
    if at < n:
        trus = knn_brute(qs, X, k=at, metric=metric)
    else:
        trus = np.broadcast_to(np.arange(n), (nq, n))
    ivf = IVF(metric, int(n**0.5), FastPQ(2))
    ivf.fit(X).build(X, n_probes=build_probes)
    recall_at = 0
    for q, tru in zip(qs, trus):
        guess = ivf.query(q, k=at, n_probes=query_probes)
        recall_at += len(set(guess) & set(tru))
    return recall_at / nq / at


@pytest.mark.parametrize("metric", ["angular", "euclidean"])
def test_monotone(metric):
    n = 4
    table = []
    for build_probes in range(1, n + 1):
        table.append([])
        for query_probes in range(1, n + 1):
            recall = compute_recall(metric, build_probes, query_probes)
            table[-1].append(recall)

    for row in table:
        print(row)

    # Test rows monotone
    for i in range(1, n):
        for j in range(n):
            assert table[i][j] >= table[i - 1][j] - 0.1

    # Test cols monotone
    for i in range(n):
        for j in range(1, n):
            assert table[i][j] >= table[i][j - 1] - 0.1

@pytest.mark.parametrize("metric", ["angular", "euclidean"])
def test_good(metric):
    n = 1000
    d = 10
    nq = 30
    at = 10
    dpb = 2
    max_probes = 10

    X = np.random.randn(n, d).astype(np.float32)
    qs = np.random.randn(nq, d).astype(np.float32)

    assert compute_recall(metric, build_probes=4, query_probes=10) >= .9
    assert compute_recall(metric, build_probes=10, query_probes=4) >= .9
