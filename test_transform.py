import random
import numpy as np

from _fast_pq import query_pq_sse
from _transform import transform_data, transform_tables


def test_simple():
    dat = [[1, 3, 7, 15]] + [[0, 0, 0, 0]] * 15
    tab = [list(range(16)) for i in range(4)]
    out = _slow_pq(np.array(dat, dtype=np.uint8), np.array(tab, dtype=np.uint8))
    assert out[0] == sum((1, 3, 7, 15))
    assert all(out[i] == 0 for i in range(1, 16))


def test_rand():
    for i in range(1, 10):
        for j in range(1, 10):
            _test_rand_inner(16 * i, 2 * j)


def _test_rand_inner(n, d):
    dat = [[random.randrange(16) for _ in range(d)] for _ in range(n)]
    # We saturate at 255, so no reason to have numbers much bigger than 1/d of that.
    tab = [[random.randrange(256 // d * 2) for _ in range(16)] for _ in range(d)]
    expected = np.array([sum(tab[j][dat[i][j]] for j in range(d)) for i in range(n)])
    expected = np.minimum(expected, 255)
    out = _slow_pq(np.array(dat, dtype=np.uint8), np.array(tab, dtype=np.uint8))
    assert np.all(expected == out)


def _slow_pq(data0, tables0):
    data = transform_data(data0)
    tables = transform_tables(tables0)
    out = np.zeros(2 * len(data), dtype=np.uint64)
    query_pq_sse(data, tables, out)
    res = out.view(np.uint8)
    assert res.shape[0] == data0.shape[0]
    return res
