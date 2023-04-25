#cython: boundscheck=False, nonecheck=False, cdivision=True, language_level=3

from libc.stdint cimport uint64_t, int64_t, uint8_t
import numpy as np
cimport numpy as np
cimport cython


cdef extern from *:
    ctypedef int byte "__int8_t"

from libcpp cimport bool

cdef extern from "immintrin.h":
    ctypedef int __m128i
    ctypedef int __m256i

    # AVX intrinsics
    __m256i _mm256_setzero_si256() nogil
    __m256i _mm256_adds_epi8(__m256i a, __m256i b) nogil
    __m256i _mm256_adds_epu8(__m256i a, __m256i b) nogil
    __m128i _mm_adds_epi8 (__m128i a, __m128i b) nogil
    __m128i _mm_adds_epu8 (__m128i a, __m128i b) nogil
    __m256i _mm256_add_epi8(__m256i a, __m256i b) nogil
    __m256i _mm256_and_si256(__m256i a, __m256i b) nogil
    __m256i _mm256_srli_epi64(__m256i a, int imm8) nogil
    __m256i _mm256_shuffle_epi8(__m256i a, __m256i b) nogil
    __m256i _mm256_set1_epi8(char a) nogil
    __m256i _mm256_set_m128i(__m128i a, __m128i b) nogil
    __m256i _mm256_cmplt_epi8(__m256i a, __m256i b) nogil
    __m256i _mm256_cmpgt_epi8(__m256i a, __m256i b) nogil
    __m256i _mm256_broadcastsi128_si256(__m128i a) nogil
    int _mm256_movemask_epi8(__m256i a) nogil
    __m256i _mm256_loadu_si256(__m256i* mem_addr) nogil
    __m128i _mm_loadu_si128 (__m128i* mem_addr) nogil
    long long _mm256_extract_epi64(__m256i a, int index) nogil
    uint64_t _mm_extract_epi64 (__m128i a, const int imm8) nogil
    long long _mm256_extracti128_si256(__m256i a, int index) nogil
    int __tzcnt_u32 (unsigned int a) nogil

    __m128i _mm_cmplt_epi8 (__m128i a, __m128i b) nogil
    __m128i _mm_add_epi8 (__m128i a, __m128i b) nogil
    __m128i _mm_add_epu8 (__m128i a, __m128i b) nogil
    int _mm_movemask_epi8 (__m128i a) nogil
    __m128i _mm_set1_epi8 (char a) nogil


cdef __m256i mm256_cmplt_epi8(__m256i a, __m256i b) nogil:
    return _mm256_cmpgt_epi8(b, a)


cpdef void estimate_pq_avx(uint64_t[:,::1] data, uint64_t[::1] tables, uint64_t[::1] out, bool signd) nogil:
    cdef:
        int i
        __m128i block_dists

    # data has shape (n//16, d) where d % 4 == 0. We read 4 64-bit numbers from data[i]
    # at a time.
    for i in range(data.shape[0]):
        block_dists = compute_block_dists_avx(&data[i, 0], data.shape[1]//4, tables, signd)
        out[2*i]   = _mm_extract_epi64(block_dists, 0)
        out[2*i+1] = _mm_extract_epi64(block_dists, 1)


cpdef void query_pq_avx(uint64_t[:,::1] data, int n, uint64_t[::1] tables, int64_t[::1] indices, int[::1] vals, bool signd, int64_t[::1] labels = None) nogil:
    cdef:
        int i, j, pos
        __m128i block_dists, top_bound, cmp_mask
        int cmp_bits, tz, bits
        uint64_t dists
        int64_t label

    top_bound = _mm_set1_epi8(vals[0])

    for i in range(data.shape[0]):
        block_dists = compute_block_dists_avx(&data[i,0], data.shape[1]//4, tables, signd)

        # Compare the computed block distances with the current largest distance. If none of the
        # new points are small enough to be added to the heap, we can just ignore the block and
        # move on.
        if signd:
            cmp_mask = _mm_cmplt_epi8(block_dists, top_bound)
        else:
            # There is no unsigned comparison in SSE, which is annoying.
            cmp_mask = _mm_cmplt_epi8(
                _mm_add_epi8 (block_dists, _mm_set1_epi8(-128)),
                _mm_add_epi8 (top_bound, _mm_set1_epi8(-128)))

        # Collect "compare mask" into a simple 16 bit integer
        cmp_bits = _mm_movemask_epi8(cmp_mask)
        # If there are any bits set in the comparison mask, process them
        if cmp_bits:
            # Handle upper and lower 64 bits of block_dists individually
            for j in range(2):
                bits = (cmp_bits >> 8*j) & 0xff
                if bits:
                    if j == 0: dists = _mm_extract_epi64(block_dists, 0)
                    if j == 1: dists = _mm_extract_epi64(block_dists, 1)
                    # pos is the index/label of the point.
                    pos = i * 16 + 8*j
                    # Iterate through the bits individually
                    while bits:
                        # Find the first 1 and shift the zeros away (tz = trailing zeros.)
                        # Alternative: "if not bits & 1: bits >>= 1; continue"
                        tz = __tzcnt_u32(bits)
                        pos, bits, dists = pos+tz, bits >> tz, dists >> 8*(tz)

                        # Ignore padding elements. Somehow it's faster to let the loop complete,
                        # Rather than "if pos >= n: break" or something like that. Maybe because
                        # of instruction pipelining?
                        if pos < n:
                            # Insert the new value into the heap
                            # TODO: What about labels than don't fit in an int?
                            label = pos if labels is None else labels[pos]
                            if signd:
                                insert(indices, vals, label, <byte>(dists & 0xff))
                            else:
                                insert(indices, vals, label, dists & 0xff)

                        pos, bits, dists = pos+1, bits >> 1, dists >> 8

            # Update bound vector to equal 16 times the largest distance in the array
            top_bound = _mm_set1_epi8(vals[0])


cdef inline __m128i compute_block_dists_avx(uint64_t* data, int block_size, uint64_t[::1] tables, bool signd) nogil:
    cdef:
        int j
        __m256i hi_table, lo_table
        __m256i block, block_masked, lo_block, hi_block, dists
        __m256i low_mask = _mm256_set1_epi8(0x0f)
        __m256i block_dists = _mm256_setzero_si256()
        __m128i lo, hi

    for j in range(block_size):
        # We eat 4 times 64 bits per cycle
        block = _mm256_loadu_si256(<__m256i*> &data[4*j])

        # Handle lower and upper nibbles
        for k in range(2):
            lo_table = _mm256_set_m128i(
                    _mm_loadu_si128(<__m128i*> &tables[8*j + 4 + 2*k]),
                    _mm_loadu_si128(<__m128i*> &tables[8*j + 2*k]))
            block_masked = block & low_mask
            dists = _mm256_shuffle_epi8(lo_table, block_masked)
            if signd:
                block_dists = _mm256_adds_epi8(block_dists, dists)
            else: block_dists = _mm256_adds_epu8(block_dists, dists)
            block >>= 4

    # Combine the two lanes
    lo = _mm256_extracti128_si256(block_dists, 0)
    hi = _mm256_extracti128_si256(block_dists, 1)
    if signd:
        return _mm_adds_epi8(lo, hi)
    else: return _mm_adds_epu8(lo, hi)


cpdef void init_heap(int64_t[::1] indices, int[::1] vals, bool signd) nogil:
    cdef:
        int K = indices.shape[0]
    if signd:
        for i in range(K):
            indices[i] = -1
            vals[i] = 127
    else:
        for i in range(K):
            indices[i] = -1
            vals[i] = 255


cpdef void insert_is(int64_t[::1] indices, int[::1] vals, int64_t i, int v) nogil:
    cdef:
        int n = indices.shape[0]
        int j = 0

    for j in range(n):
        if i == indices[j]:
            return
    j = 0

    while j+1 != n and vals[j+1] > v:
        indices[j], vals[j] = indices [j+1], vals[j+1]
        j += 1
    indices[j], vals[j] = i, v


cpdef void insert(int64_t[::1] indices, int[::1] vals, int64_t i, int v) nogil:
    cdef:
        int n = indices.shape[0]
        int j = 0
        int nxt, l, r, nxt_val

    for j in range(n):
        if i == indices[j]:
            return
    j = 0

    while True:
        nxt, nxt_val = j, v
        l, r = 2*j+1, 2*j+2
        if l < n and vals[l] > nxt_val:
            nxt, nxt_val = l, vals[l]
        if r < n and vals[r] > nxt_val:
            nxt, nxt_val = r, vals[r]
        if nxt == j:
            vals[j], indices[j] = v, i
            break
        vals[j], indices[j] = vals[nxt], indices[nxt]
        j = nxt



