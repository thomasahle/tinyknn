import random
import numpy as np
from functools import reduce

from fast_pq import pad
from fast_pq._fast_pq import estimate_pq_sse
from fast_pq._transform import transform_data, transform_tables


def test_pad():
    np.random.seed(12)
    for _ in range(10):
        for d in range(1, 4):
            for m in range(1, 4):
                shape = [np.random.randint(1, 10) for _ in range(d)]
                ar = np.zeros(shape)
                shape2 = pad(ar, [m] * d).shape
                assert all(
                    s1 <= s2 < s1 + m and s2 % m == 0 for s1, s2 in zip(shape, shape2)
                )


def test_simple():
    dat = [[1, 3, 7, 15]] + [[0, 0, 0, 0]] * 15
    tab = [list(range(16)) for i in range(4)]
    out = _slow_pq(
        np.array(dat, dtype=np.uint8), np.array(tab, dtype=np.uint8), signed=False
    )
    assert out[0] == sum((1, 3, 7, 15))
    assert all(out[i] == 0 for i in range(1, 16))


def test_rand():
    for i in range(1, 10):
        for j in range(1, 10):
            _test_rand_inner_signed(16 * i, 2 * j)
            _test_rand_inner_unsigned(16 * i, 2 * j)


def _test_rand_inner_unsigned(n, d):
    dat = [[random.randrange(16) for _ in range(d)] for _ in range(n)]
    # We saturate at 255, so no reason to have numbers much bigger than 1/d of that.
    tab = [[random.randrange(256 // d * 2) for _ in range(16)] for _ in range(d)]
    expected = np.array([sum(tab[j][dat[i][j]] for j in range(d)) for i in range(n)])
    expected = np.minimum(expected, 255)
    out = _slow_pq(np.array(dat, dtype=np.uint8), np.array(tab, dtype=np.uint8), False)
    assert np.all(expected == out)


def sat8_add(x, y):
    if x + y > 127:
        return 127
    if x + y < -128:
        return -128
    return x + y


def _test_rand_inner_signed(n, d):
    dat = [[random.randrange(16) for _ in range(d)] for _ in range(n)]
    # We saturate at 127, so no reason to have numbers much bigger than 1/sqrt(d) of that.
    top = 127 // d**0.5
    tab = [[random.randrange(-top, top) for _ in range(16)] for _ in range(d)]
    # For unsigned addition saturation is associative, but for signed we have
    # to keep track of the stauration at every step.
    expected = np.array(
        [reduce(sat8_add, (tab[j][dat[i][j]] for j in range(d))) for i in range(n)]
    )
    out = _slow_pq(np.array(dat, dtype=np.uint8), np.array(tab, dtype=np.uint8), True)
    assert np.all(expected == out.astype(np.int8))


def _slow_pq(data0, tables0, signed):
    data = transform_data(data0)
    tables = transform_tables(tables0)
    out = np.zeros(2 * len(data), dtype=np.uint64)
    estimate_pq_sse(data, tables, out, signed)
    print(out)
    res = out.view(np.uint8)
    assert res.shape[0] == data0.shape[0]
    return res
