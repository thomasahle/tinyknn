import numpy as np
import time
from contextlib import contextmanager


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


@contextmanager
def timer(verbose, text):
    if verbose:
        print(text)
        start = time.time()
    yield
    if verbose:
        print(f"Took {time.time() - start:.1f}s")


def cdist(X, Y, chunk=100):
    """
    Computes the squared Euclidean distances between two sets of points X and Y.
    Returns R st. R[i,j] = dist(X_i, Y_j)^2.
    Equivalent to scipy.spatial.distance.cdist.
    """
    # This is how sklearn computes row norms. It
    # %timeit np.linalg.norm(x, axis=1)
    # 486 ms ± 6.33 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # %timeit np.einsum('ij,ij->i',x,x)
    # 62.9 ms ± 2.35 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    # %timeit (x*x).sum(axis=1)
    # 495 ms ± 6.71 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    nx = np.einsum("ij,ij->i", X, X)
    ny = np.einsum("ij,ij->i", Y, Y)
    res = np.zeros((nx.size, ny.size))
    for i in range(0, nx.size, chunk):
        res[i : i + chunk] = nx[i : i + chunk, None] + ny
        res[i : i + chunk] -= 2 * X[i : i + chunk] @ Y.T
    return res


def brute(X, Y, k, metric="euclidean", chunk=100):
    """
    Computes the k-nearest neighbors for each point in X based on the squared Euclidean distances
    between X and Y.
    """
    assert k <= Y.shape[0], f"Can't find knn with {k=} and {Y.shape[0]} targets."
    if metric == "angular":
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    elif metric not in ["angular", "euclidean"]:
        raise ValueError(f"Metric not supported: {metric}")
    n = X.shape[0]
    res = np.zeros((n, k), dtype=int)
    Ynorm2 = np.einsum("ij,ij->i", Y, Y)
    for i in range(0, n, chunk):
        Xchunk = X[i : i + chunk]
        Xnorm2 = np.einsum("ij,ij->i", Xchunk, Xchunk)
        part = Xnorm2[:, None] + Ynorm2[None] - 2 * Xchunk @ Y.T
        res[i : i + chunk] = bottom_k_2d(part, k)
    return res


def brute1(x, Y, k):
    """
    Computes the k-nearest neighbors for each point in X based on the squared Euclidean distances
    between X and Y.
    """
    diff = Y - x
    dists = np.einsum("ij,ij->i", diff, diff)
    return bottom_k(dists, k)


def group_data_by_indices(X, indices, k):
    """
    Given a 2D array `X` of shape (N, d), a 2D array `indices` of shape (N, c) with integers
    in [0, k), and an integer `k`, return a list `parts` of k arrays, such that all rows X[i]
    are in parts[indices[i, j]] for some j.
    Args:
    X (np.ndarray): A 2D numpy array of shape (N, d) containing the data points.
    indices (np.ndarray): A 2D numpy array of shape (N, c) containing integers in the range [0, k).
    k (int): The number of groups.

    Returns:
    list: A list of k numpy arrays, where each array contains the rows of X that belong to the corresponding group.
    """
    # Initialize an empty list to store the data points for each group
    assert 0 <= np.min(indices) and np.max(indices) < k
    parts = [[] for _ in range(k)]
    ids = [[] for _ in range(k)]

    # Iterate over each column in indices
    for i in range(indices.shape[1]):
        # Get the current column of indices
        col = indices[:, i]

        # Sort X by the current column of indices
        sorted_indices = np.argsort(col)
        sorted_X = X[sorted_indices]

        # Compute the length of the runs in the current column of indices (sorted)
        sorted_indices_unique, run_lengths = np.unique(
            col[sorted_indices], return_counts=True
        )
        # Use a loop only over j = 0 to k - 1 that selects the runs of X and adds them to the corresponding group in parts
        start = 0
        for i, run_length in enumerate(run_lengths):
            end = start + run_length
            parts[sorted_indices_unique[i]].append(sorted_X[start:end])
            ids[sorted_indices_unique[i]].append(sorted_indices[start:end])
            start = end

    # Check for empty lists
    for part, id_list in zip(parts, ids):
        if len(part) == 0:
            assert len(id_list) == 0
            part.append(np.empty((0, X.shape[1])))
            id_list.append(np.empty(0))

    ids = [np.hstack(id_list) for id_list in ids]
    parts = [np.vstack(part) for part in parts]
    return parts, ids
