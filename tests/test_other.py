import numpy as np
from fast_pq._fast_pq import insert as heap_insert


def test_heap():
    np.random.seed(13)
    for n in range(1, 10):
        _test_sequence(list(range(n)))
        _test_sequence(list(reversed(range(n))))
        for _ in range(3):
            _test_sequence([np.random.randint(n) for _ in range(n)])


def _test_sequence(vs):
    values = np.full(len(vs), max(vs) + 1, dtype=np.int32)
    indices = np.zeros(len(vs), dtype=np.int32)
    verify_max_heap(values, 0)
    for i, v in enumerate(vs):
        heap_insert(indices, values, i, v)
        assert v in values
        verify_max_heap(values, 0)


def verify_max_heap(values, root):
    n = len(values)
    l, r = 2 * root + 1, 2 * root + 2
    if l < n:
        assert values[l] <= values[root]
        verify_max_heap(values, l)
    if r < n:
        assert values[r] <= values[root]
        verify_max_heap(values, r)
