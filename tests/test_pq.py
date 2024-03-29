import numpy as np
import pytest
from itertools import product

from tinyknn._fast_pq import estimate_pq_sse, query_pq_sse, init_heap
from tinyknn._fast_pq_avx import estimate_pq_avx, query_pq_avx
from tinyknn._transform import transform_tables, transform_data
from tinyknn import FastPQ, knn_brute

np.random.seed(10)

@pytest.mark.parametrize(
        "n, d, signed, simd_method",
        product([16, 32], [4, 8], [True, False], ['sse', 'avx'])
    )
def test_estimate_pq_simd(n, d, signed, simd_method):
    data = np.random.randint(0, 16, size=(n, d), dtype=np.uint8)
    tables = np.random.randint(0, 256, size=(d, 16), dtype=np.uint8)
    out = np.zeros(shape=(n//8,), dtype=np.uint64)
    byte_type = np.int8 if signed else np.uint8

    # Call the Cython function
    simd_func = estimate_pq_avx if simd_method == 'avx' else estimate_pq_sse
    simd_func(transform_data(data), transform_tables(tables), out, signed)
    res = out.view(byte_type)
    assert len(res) == n

    # Compute the expected output
    expected = np.zeros(shape=(n,), dtype=byte_type)
    tables = tables.view(byte_type)
    for i, row in enumerate(data):
        dist = 0
        if simd_method == 'sse':
            for j, ptr in enumerate(row):
                dist += int(tables[j][ptr])
                # Clip values to simulate saturated addition
                dist = np.clip(dist, -128, 127) if signed else np.clip(dist, 0, 255)
        # AVX ordering is a bit odd right now, which affects saturation
        elif simd_method == 'avx':
            dist0, dist1 = 0, 0
            for j, ptr in enumerate(row):
                if j & 2 == 0:
                    dist0 += int(tables[j][ptr])
                    dist0 = np.clip(dist0, -128, 127) if signed else np.clip(dist0, 0, 255)
                else:
                    dist1 += int(tables[j][ptr])
                    dist1 = np.clip(dist1, -128, 127) if signed else np.clip(dist1, 0, 255)
            dist = dist0 + dist1
            dist = np.clip(dist, -128, 127) if signed else np.clip(dist, 0, 255)

        expected[i] = dist

    np.testing.assert_array_equal(res, expected)


@pytest.mark.parametrize(
    "i, method, signed, use_kmeans",
    product(range(1, 5), ["argpartition", "top"], [True, False], [True, False]),
)
def test_recall(i, method, signed, use_kmeans):
    n = np.random.randint(16 * i, 16 * (i + 1))
    _test_recall_inner(n, 8 * i, 100, 2, method, signed, use_kmeans)


def _test_recall_inner(n, d, k, dpb, method, signed, use_kmeans):
    X = np.random.randn(n, d).astype(np.float32)
    qs = np.random.randn(k, d).astype(np.float32)
    trus = knn_brute(qs, X, k=1)[:, 0]

    pq = FastPQ(dims_per_block=dpb, use_kmeans=use_kmeans)
    data = pq.fit_transform(X)
    recall_at_10 = 0
    for q, tru in zip(qs, trus):
        dtable = pq.distance_table(q) if signed else pq.udistance_table(q)
        if method == "argpartition":
            top10 = dtable.estimate_distances(data).argpartition(10)[:10]
        elif method == "top":
            top10 = dtable.top(data, X, 10)
        if tru in top10:
            recall_at_10 += 1
    recall_at_10 /= k
    assert recall_at_10 > 0.8


@pytest.mark.filterwarnings("ignore:Number of distinct clusters")
@pytest.mark.parametrize(
    "n, dpb, signed", product(tuple(range(1, 10)) + (20, 30, 50), [1, 2], [True, False])
)
def test_topk(n, dpb, signed):
    _test_topk_inner(n, 3, 11, dpb, signed)


@pytest.mark.filterwarnings("ignore:Number of distinct clusters")
@pytest.mark.parametrize("signed", [True, False])
def test_topk_0(signed):
    with pytest.raises(AssertionError):
        _test_topk_inner(0, 3, 11, 2, signed)


def test_fit_transform():
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
        est = est[:n]  # Remove padding

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
