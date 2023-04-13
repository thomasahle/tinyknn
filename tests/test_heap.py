import pytest
import numpy as np
import heapq

np.random.seed(10)

from fast_pq._fast_pq import estimate_pq_sse
from fast_pq._fast_pq import insert, init_heap

class Heap:
    def __init__(self, size):
        self.indices = np.empty((size,), dtype=np.int64)
        self.vals = np.empty((size,), dtype=np.int32)
        init_heap(self.indices, self.vals, signd=True)

    def insert(self, i, v):
        if v < self.peek():
            insert(self.indices, self.vals, i, v)

    def peek(self):
        return self.vals[0]


def test_heap_init():
    heap = Heap(3)
    np.testing.assert_array_equal(heap.indices, np.array([-1] * 3, dtype=np.int64))
    np.testing.assert_array_equal(heap.vals, np.array([127] * 3, dtype=np.int32))


def test_heap_insert_single_element():
    heap = Heap(1)
    heap.insert(1, 10)

    np.testing.assert_array_equal(heap.indices, np.array([1], dtype=np.int64))
    np.testing.assert_array_equal(heap.vals, np.array([10], dtype=np.int32))


def test_heap_insert_two_elements():
    heap = Heap(2)
    heap.insert(1, 10)
    np.testing.assert_array_equal(heap.indices, np.array([-1, 1], dtype=np.int64))
    np.testing.assert_array_equal(heap.vals, np.array([127, 10], dtype=np.int32))


def test_heap_unique():
    heap = Heap(2)
    heap.insert(1, 10)
    heap.insert(1, 10)
    np.testing.assert_array_equal(heap.indices, np.array([-1, 1], dtype=np.int64))
    np.testing.assert_array_equal(heap.vals, np.array([127, 10], dtype=np.int32))


def test_random():
    heap = Heap(10)
    pyheap = [(-127, -1)] * 10
    for t in range(1000):
        top_pyheap = -pyheap[0][0]
        assert top_pyheap == heap.peek()

        v = np.random.randint(10000 // (t+1))
        heap.insert(t, v)
        if v < top_pyheap:
            heapq.heappop(pyheap)
            heapq.heappush(pyheap, (-v, t))
        assert set(heap.vals) == set(-vi for vi, _ in pyheap)


def test_heap():
    np.random.seed(13)
    for n in range(1, 10):
        _test_sequence(list(range(n)))
        _test_sequence(list(reversed(range(n))))
        for _ in range(3):
            _test_sequence([np.random.randint(n) for _ in range(n)])


def _test_sequence(vs):
    heap = Heap(len(vs))
    verify_max_heap_property(heap.vals, 0)
    for i, v in enumerate(vs):
        heap.insert(i, v)
        assert v in heap.vals
        assert i in heap.indices
        verify_max_heap_property(heap.vals, 0)


def verify_max_heap_property(values, root):
    n = len(values)
    l, r = 2 * root + 1, 2 * root + 2
    if l < n:
        assert values[l] <= values[root]
        verify_max_heap_property(values, l)
    if r < n:
        assert values[r] <= values[root]
        verify_max_heap_property(values, r)
