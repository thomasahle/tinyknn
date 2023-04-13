#cython: boundscheck=False, nonecheck=False, cdivision=True, language_level=3

# For each block (say 50) there is one code in data (say data[i,j]) of 4 bits.
# Actually, I suppose that means data needs to be packed, so it has dimension only 25.
# And we need to look the code up in a distance table for the block.
# There are 16 8-bit distances for each block. That's 128 bits.
# Probably then table should just be a 50*16 byte array.
# Hm. Maybe this is all wrong, since shuffle takes the 16 blocks and look them up in the same table.
# Instead I should be processing 16 different vectors, all of which need to look up in the same
# table.
# Hvorfor ikke bare transpose hele dataen? Så skal hver del af table kun loades en gang.
# Så kan table f.eks. være en blocks * 16 byte array.

# Standard format
#    a0 a1 a2 ... ad
#    b0 b1 b2 ... bd <- 4 bits
# Format as in Quick ADC:
#    a0 b0 c0 ... p0 <- 16'th letter. Whole row is 64 bit
#    a1 b1 c1 ... p1
#    ...
#    ad bd cd ... pd
#    q0 r0 s0 ...    <- New chunk
# Ok, so maybe the format is just uint64[n/16, d].
# This will also support residual codes, if we are so inclined.
# Just make two different tables

# If the block_size (d) is quite big, it might be beneficial to split
# into column chunks and run this function multiple times, then combine
# the results. That would (hopefully) allow the table to be kept in registers.
# It would also help prevent lane saturation.

# Format for table:
# uint128[2*d]
# We can use pshufb to do 16 1byte lookups in one operation.
# See https://arxiv.org/pdf/1812.09162.pdf

cdef extern from *:
    ctypedef int uint64_t "__uint64_t"
    ctypedef int int64_t "__int64_t"
    ctypedef int uint128_t "__uint128_t"
    ctypedef int byte "__int8_t"

from libcpp cimport bool

cdef extern from "immintrin.h":
    ctypedef int  __m128i
    ctypedef int  __m256i
    # Shuffle packed 8-bit integers in a according to shuffle control mask in
    # the corresponding 8-bit element of b, and store the results in dst.
    # Note that we only use the last 4 bits of each byte in b.
    # Furthermore, the first bit in each byte can be used to get a zero, rather
    # than a lookup. Thus we need to mask the low 4bits when we use this.
    __m128i _mm_shuffle_epi8 (__m128i a, __m128i b) nogil
    # Shuffle 8-bit integers in a within 128-bit lanes according to shuffle
    # control mask in the corresponding 8-bit element of b, and store the
    # results in dst.
    __m256i _mm256_shuffle_epi8 (__m256i a, __m256i b) nogil
    # Load 128-bits of integer data from memory into dst. mem_addr does not
    # need to be aligned on any particular boundary.
    #__m128i _mm_loadu_si128 (__m128i const* mem_addr) nogil
    __m128i _mm_add_epi8 (__m128i a, __m128i b) nogil
    # Maybe we need saturating addition?
    __m256i _mm256_adds_epu8(__m256i s1, __m256i s2) nogil
    # Same with SSE
    __m128i _mm_adds_epu8(__m128i s1, __m128i s2) nogil
    # Add packed signed 8-bit integers in a and b using saturation, and store the results in dst.
    # Signed giver mening med IP, men måske ikke så meget med afstande...
    # Jeg kunne selvfølgelig bare starte på afstand 0...
    __m128i _mm_adds_epi8 (__m128i a, __m128i b) nogil
    __m128i _mm_set_epi64x (uint64_t e1, uint64_t e0) nogil
    # Load 128-bits of integer data from memory into dst. mem_addr does not need
    # to be aligned on any particular boundary.
    __m128i _mm_loadu_si128 (__m128i* mem_addr) nogil
    # Load 128-bits of integer data from unaligned memory into dst. This
    # intrinsic may perform better than _mm_loadu_si128 when the data crosses a
    # cache line boundary.
    __m128i _mm_lddqu_si128 (__m128i* mem_addr) nogil
    # Copy the lower 64-bit integer in a to dst.
    uint64_t _mm_cvtsi128_si64x (__m128i a) nogil
    # Extract a 64-bit integer from a, selected with imm8, and store the result in dst.
    uint64_t _mm_extract_epi64 (__m128i a, const int imm8) nogil

    __m128i _mm_srli_epi64 (__m128i a, int imm8) nogil
    __m128i _mm_and_si128 (__m128i a, __m128i b) nogil
    __m128i _mm_setzero_si128 () nogil

    # Broadcast 8-bit integer a to all elements of dst. This intrinsic may
    # generate vpbroadcastb.
    __m128i _mm_set1_epi8 (char a) nogil
    # Compare packed signed 8-bit integers in a and b for less-than, and store
    # the results in dst. Note: This intrinsic emits the pcmpgtb instruction
    # with the order of the operands switched.
    __m128i _mm_cmplt_epi8 (__m128i a, __m128i b) nogil
    # Create mask from the most significant bit of each 8-bit element in a,
    # and store the result in dst.
    int _mm_movemask_epi8 (__m128i a) nogil
    # Count the number of trailing zero bits in unsigned 64-bit integer a, and return that count in dst.
    int __tzcnt_u32 (unsigned int a) nogil


cpdef void estimate_pq_sse(uint64_t[:,::1] data, uint64_t[::1] tables,
                           uint64_t[::1] out, bool signd) nogil:
    cdef:
        int i
        __m128i block_dists

    for i in range(data.shape[0]):
        # We can't just pass data[i] here, since that would allocate a new slice
        block_dists = compute_block_dists(&data[i, 0], data.shape[1]//2, tables, signd)
        out[2*i]   = _mm_extract_epi64(block_dists, 0)              # out[2i] = block_dists[0:8]
        out[2*i+1] = _mm_extract_epi64(block_dists, 1)              # out[2i] = block_dists[8:16]


cpdef void query_pq_sse(uint64_t[:,::1] data, int n, uint64_t[::1] tables,
                        int64_t[::1] indices, int[::1] vals, bool signd,
                        int64_t[::1] labels = None,
                        ) nogil:
    '''
    Given a N x D dataset quantized into byte-sized chunks,
    looks up each value in a table and outputs into `out`.

    Parameters
    ----------
    data : 2D memoryview of uint64_t
        The quantized dataset. Each 64bit int represents a (four bit) column in
        a chunk of 16 rows.
        See _transform.py for the exact format used, borrowed from Quick ADC.

    n : int
        The actual length of the data, without the padding. Used to avoid including
        padding elements in the final output.

    tables : 1D memoryview of uint64_t
        The lookup tables. Each pair of 64bit ints represent a uint8[16] lookup table.
        See _transform.py for the exact format used.

    indices : 1D memoryview of int
        The output indices.

    vals : 1D memoryview of int
        The output values.

    signd : bool
        Determines if the distance is signed or unsigned.
    '''
    cdef:
        int i, j, pos
        __m128i block_dists, top_bound, cmp_mask
        int cmp_bits, tz, bits
        uint64_t dists
        int64_t label

    top_bound = _mm_set1_epi8(vals[0])

    # Iterate through the data chunks
    for i in range(data.shape[0]):
        # Compute the (8 bit) distances from the query to each of the points in the 16 point block
        block_dists = compute_block_dists(&data[i,0], data.shape[1]//2, tables, signd)

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
                            # How many bits are even in an int?
                            label = pos if labels is None else labels[pos]
                            if signd:
                                insert(indices, vals, label, <byte>(dists & 0xff))
                            else:
                                insert(indices, vals, label, dists & 0xff)

                        pos, bits, dists = pos+1, bits >> 1, dists >> 8

            # Update bound vector to equal 16 times the largest distance in the array
            top_bound = _mm_set1_epi8(vals[0])


cdef inline __m128i compute_block_dists(uint64_t* data, int block_size,
                                        uint64_t[::1] tables, bool signd) nogil:
    cdef:
        int j
        __m128i block_dists
        __m128i hi_table, lo_table
        __m128i block, block_masked, lo_block, hi_block, dists
        __m128i low_mask = _mm_set1_epi8(0x0f)

    block_dists = _mm_setzero_si128()
    # We read two 64bit chunks at a time
    for j in range(block_size):
        # Do we need to use _mm_loadu_si128 or _mm_lddqu_si128 to load data?
        # I think they are only needed when data can be unaligned, but if we
        # use a numpy uint128, won't it be aligned?
        block = _mm_loadu_si128(<__m128i*> &data[2*j])
        # Low comps
        lo_table = _mm_loadu_si128(<__m128i*> &tables[4*j])     # load 128 bits
        block_masked = _mm_and_si128(block, low_mask);          # & low_mask
        dists = _mm_shuffle_epi8(lo_table, block_masked)        # table lookup
        # Hopefully this will be specialized by the compiler
        if signd:
            block_dists = _mm_adds_epi8(block_dists, dists)     # block_dists += dists
        else: block_dists = _mm_adds_epu8(block_dists, dists)
        # High comps
        hi_table = _mm_loadu_si128(<__m128i*> &tables[4*j+2])   # load 128 bits
        block_masked = _mm_srli_epi64(block, 4)                 # >> 4
        block_masked = _mm_and_si128(block_masked, low_mask)    # & low_mask
        dists = _mm_shuffle_epi8(hi_table, block_masked)        # table lookup
        if signd:
            block_dists = _mm_adds_epi8(block_dists, dists)     # block_dists += dists
        else: block_dists = _mm_adds_epu8(block_dists, dists)
    return block_dists


# Maybe it's silly to have this is cython, rather than just use numpy.
cpdef void init_heap(int64_t[::1] indices, int[::1] vals, bool signd) nogil:
    cdef:
        int K = indices.shape[0]
    # Even though we have an int array, we have to use values that fit in 8 bits,
    # since we are going to broadcast them to an __m128i.
    if signd:
        for i in range(K):
            indices[i] = -1
            vals[i] = 127
    else:
        for i in range(K):
            indices[i] = -1
            vals[i] = 255


# Using insertion sort. Also an option.
cpdef void insert_is(int64_t[::1] indices, int[::1] vals, int64_t i, int v) nogil:
    cdef:
        int n = indices.shape[0]
        int j = 0
    # First see if we are already in the array
    for j in range(n):
        if i == indices[j]:
            return
    j = 0
    # We assume that vals[0] > v, or this function wouldn't have been called
    # in the first place.
    while j+1 != n and vals[j+1] > v:
        # Shift left
        indices[j], vals[j] = indices[j+1], vals[j+1]
        j += 1
    indices[j], vals[j] = i, v


cpdef void insert_old(int64_t[::1] indices, int[::1] vals, int64_t i, int v) nogil:
    ''' Insert (i,v) into the list, which is assumed ordered by vals '''
    # We need to easily be able to identify the largest element in the heap,
    # since that's the one we are kicking out. Thus this is a max-heap with
    # the largest value at 0.
    cdef:
        int n = indices.shape[0]
        int j = 0
        int nxt, l, r

    # First see if we are already in the array
    for j in range(n):
        if i == indices[j]:
            return
    j = 0

    # Insert the new value at the top, replacing the old furthest point.
    # We assume it's given that our value is at most as large as that old point.
    indices[0], vals[0] = i, v
    # Swap with the children until we are at least as large as both of them,
    # or we reach the end of the array.
    while True:
        # Our current candidate for the next node: don't change.
        nxt = j
        l, r = 2*j+1, 2*j+2
        # Swap with the largest of the two children, assuming
        # then are not out of bounds.
        if l < n and vals[l] > vals[nxt]: nxt = l
        if r < n and vals[r] > vals[nxt]: nxt = r
        # If we didn't pick any of the children, we are done.
        if nxt == j:
            break
        # Swap with the child.
        vals[nxt], vals[j] = vals[j], vals[nxt]
        indices[nxt], indices[j] = indices[j], indices[nxt]
        j = nxt


cpdef void insert(int64_t[::1] indices, int[::1] vals, int64_t i, int v) nogil:
    ''' Insert (i,v) into the list, which is assumed ordered by vals '''
    # We need to easily be able to identify the largest element in the heap,
    # since that's the one we are kicking out. Thus this is a max-heap with
    # the largest value at 0.
    cdef:
        int n = indices.shape[0]
        int j = 0
        int nxt, l, r, nxt_val

    # First see if we are already in the array
    for j in range(n):
        if i == indices[j]:
            return
    j = 0

    # Swap values up until both are smaller than v, or we reach the end of the array.
    while True:
        # Our current candidate for the next node: don't change.
        nxt, nxt_val = j, v
        l, r = 2*j+1, 2*j+2
        # Swap with the largest of the two children, assuming
        # then are not out of bounds.
        if l < n and vals[l] > nxt_val:
            nxt, nxt_val = l, vals[l]
        if r < n and vals[r] > nxt_val:
            nxt, nxt_val = r, vals[r]
        # If we didn't pick any of the children, we are done.
        if nxt == j:
            vals[j], indices[j] = v, i
            break
        # Move the value up
        j, vals[j], indices[j] = nxt, vals[nxt], indices[nxt]



# Maybe it doesn't make sense to do a real insertion sort, since we have to break
# up values across their boundary all the time when shifting.
# Instead we could relax the requirement, and say that we dont' need to be sorted
# in each "block"?
# void simd_insert_8bit(__m128i* arr, size_t num_blocks, uint8_t value) {
#     __m128i xmm_value = _mm_set1_epi8(value);
#     bool inserted = false;
# 
#     for (size_t i = 0; i < num_blocks; ++i) {
#         __m128i xmm_current = arr[i];
#         __m128i xmm_less_mask = _mm_cmplt_epi8(xmm_current, xmm_value);
# 
#         if not inserted and _mm_movemask_epi8(xmm_less_mask):
#             # The value has not been inserted yet and there is a position to insert it
#             xmm_current = _mm_alignr_epi8(xmm_value, xmm_current, 15);
#             inserted = True;
#         elif inserted:
#             # The value has already been inserted, shift the remaining elements
#             __m128i xmm_prev = arr[i - 1];
#             xmm_current = _mm_alignr_epi8(xmm_current, xmm_prev, 15);
#         arr[i] = xmm_current;
#     }
# }
