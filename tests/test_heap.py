import pytest
import numpy as np
import heapq

np.random.seed(10)

from fast_pq._fast_pq import estimate_pq_sse
from fast_pq._fast_pq import insert, init_heap

class Heap:
    def __init__(self, size):
        self.indices = np.empty((size,), dtype=np.int32)
        self.vals = np.empty((size,), dtype=np.int32)
        init_heap(self.indices, self.vals, signd=True)

    def insert(self, i, v):
        if v < self.peek():
            insert(self.indices, self.vals, i, v)

    def peek(self):
        return self.vals[0]

def test_heap_init():
    heap = Heap(3)
    np.testing.assert_array_equal(heap.indices, np.array([-1] * 3, dtype=np.int32))
    np.testing.assert_array_equal(heap.vals, np.array([127] * 3, dtype=np.int32))

def test_heap_insert_single_element():
    heap = Heap(1)
    heap.insert(1, 10)

    np.testing.assert_array_equal(heap.indices, np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(heap.vals, np.array([10], dtype=np.int32))

def test_heap_insert_two_elements():
    heap = Heap(2)
    heap.insert(1, 10)
    np.testing.assert_array_equal(heap.indices, np.array([-1, 1], dtype=np.int32))
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
        assert heap.peek() == -pyheap[0][0]

