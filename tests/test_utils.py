import pytest
import numpy as np
from itertools import product

from fast_pq import cdist, knn_brute, group_data_by_indices


np.random.seed(10)


def test_cdist():
    n1, n2, d = 10, 8, 5
    for chunk in [1, 10, 100]:
        X = np.random.randn(n1, d)
        Y = np.random.randn(n2, d)
        dists = cdist(X, Y, chunk=chunk)
        for i in range(n1):
            for j in range(n2):
                tru_dist = np.sum((X[i] - Y[j]) ** 2)
                assert np.isclose(dists[i, j], tru_dist)


@pytest.mark.parametrize(
    "n1, n2, d, k",
    product([40], [28], [5], [0, 1, 10, 28])
)
def test_brute(n1, n2, d, k):
    X = np.random.randn(n1, d)
    Y = np.random.randn(n2, d)
    if k < n2:
        expected = cdist(X, Y).argpartition(axis=1, kth=k)[:, :k]
    else:
        expected = np.broadcast_to(np.arange(n2), (n1, n2))
    best = knn_brute(X, Y, k)
    assert np.all(np.sort(expected) == np.sort(best))


def test_angular():
    n1, n2, d = 40, 28, 5
    X = np.random.randn(n1, d)
    Y = np.random.randn(n2, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    # Ordering between euclidean and angular is the same for normalized vectors
    angular = knn_brute(X, Y, 10, metric="angular")
    euclidean = knn_brute(X, Y, 10, metric="euclidean")
    assert np.all(np.sort(angular) == np.sort(euclidean))


def test_group_data_by_indices():
    N, d, c, k = 100, 5, 6, 3
    X = np.random.rand(N, d)
    Q = np.random.randn(c, d)
    indices = np.argpartition(-X @ Q.T, k, axis=1)[:, :k]

    # Using the group_data_by_indices function
    parts, _ = group_data_by_indices(X, indices, c)

    # Using the alternative method with masks
    mask_parts = []
    for i in range(c):
        mask = np.any(indices == i, axis=1)
        mask_parts.append(X[mask])

    # Compare the results
    for i in range(c):
        # Sort so we can compare
        A = parts[i]
        sorted_A = A[np.lexsort(A.T), :]
        B = mask_parts[i]
        sorted_B = B[np.lexsort(B.T), :]
        np.testing.assert_allclose(sorted_A, sorted_B)
