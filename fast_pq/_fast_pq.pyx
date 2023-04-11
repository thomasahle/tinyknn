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


cpdef void query_pq_sse(uint64_t[:,::1] data, uint64_t[::1] tables, int[::1] indices,
                        int[::1] vals, bool signd) nogil:
    ''' Given a N x D dataset quantized into byte sizes chunks,
        looks up each value in a table out outputs into `out`. '''
    cdef:
        int i, j
        __m128i block_dists, top_bound, cmp_mask
        int cmp_bits, cmp_low, cmp_high, pos, tz, bits
        uint64_t dists

    # Initialize "heap". K = is the number of top values we want
    if signd:
        for i in range(indices.shape[0]):
            indices[i] = -1
            vals[i] = 0x7f
        top_bound = _mm_set1_epi8(0x7f)
    else:
        for i in range(indices.shape[0]):
            indices[i] = -1
            vals[i] = 0xff
        top_bound = _mm_set1_epi8(0xff)

    for i in range(data.shape[0]):
        block_dists = compute_block_dists(&data[i,0], data.shape[1]//2, tables, signd)
        if signd:
            cmp_mask = _mm_cmplt_epi8(block_dists, top_bound)
        else:
            # There is no unsigned comparison in SSE, which is annoying.
            cmp_mask = _mm_cmplt_epi8(
                _mm_add_epi8 (block_dists, _mm_set1_epi8(-128)),
                _mm_add_epi8 (top_bound, _mm_set1_epi8(-128)))
        cmp_bits = _mm_movemask_epi8(cmp_mask)
        if cmp_bits:
            #print('cmp_bits', bin(cmp_bits))
            for j in range(2):
                bits = (cmp_bits >> 8*j) & 0xff
                if bits:
                    if j == 0: dists = _mm_extract_epi64(block_dists, 0)
                    if j == 1: dists = _mm_extract_epi64(block_dists, 1)
                    pos = i * 16 + 8*j
                    while bits:
                        tz = __tzcnt_u32(bits)
                        #print(pos, bin(dists), bin(bits), tz)
                        pos, bits, dists = pos+tz, bits >> tz, dists >> 8*(tz)
                        #print('insert', pos, signd, (dists&0xff), <byte>(dists & 0xff))
                        if signd:
                            insert(indices, vals, pos, <byte>(dists & 0xff))
                        else:
                            insert(indices, vals, pos, dists & 0xff)
                        #print(list(vals))
                        pos, bits, dists = pos+1, bits >> 1, dists >> 8
            # Update bound vector to equal 16 times the largest distance in the array
            # top_bound = _mm_set1_epi8(vals[-1]) # With insertion sort, the max is at the end
            top_bound = _mm_set1_epi8(vals[0]) # If we use the heap, the max is at the start


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


cdef void insert_old(int[::1] indices, int[::1] vals, int i, int v) nogil:
    ''' Insert (i,v) into the list, which is assumed ordered by vals '''
    for j in range(indices.shape[0]):
        # Insert the new value at the found location, then continue "recursively"
        #if vals[j] > v or indices[j] == -1:
        if vals[j] > v:
            vals[j], v = v, vals[j]
            indices[j], i = i, indices[j]


cdef void insert_old2(int[::1] indices, int[::1] vals, int i, int v) nogil:
    ''' Insert (i,v) into the list, which is assumed ordered by vals '''
    # We know we are going to be somewhere in the array, so we just
    # set ourselves at the right end, then swap until we are in the
    # right position.
    cdef int j = indices.shape[0]-1
    indices[j], vals[j] = i, v
    while j != 0 and vals[j-1] > vals[j]:
        vals[j-1], vals[j] = vals[j], vals[j-1]
        indices[j-1], indices[j] = indices[j], indices[j-1]
        j -= 1


cdef void insert_insort(int[::1] indices, int[::1] vals, int i, int v) nogil:
    ''' Insert (i,v) into the list, which is assumed ordered by vals '''
    # It seems silly to do all that swappnig. Rather just move things
    # down and then eventually insert ourselves.
    # This is still using that we know we belong at least at the end,
    # since we always insert ourselves somewhere eventually.
    cdef int j = indices.shape[0]-1
    while j != 0 and vals[j-1] > v:
        indices[j], vals[j] = indices[j-1], vals[j-1]
        j -= 1
    indices[j], vals[j] = i, v


# We use cpdef so we can unittest
cpdef void insert(int[::1] indices, int[::1] vals, int i, int v) nogil:
    ''' Insert (i,v) into the list, which is assumed ordered by vals '''
    # We need to easily be able to identify the largest element in the heap,
    # since that's the one we are kicking out. Thus this is a max-heap with
    # the largest value at 0.
    cdef:
        int n = indices.shape[0]
        int j = 0
        int nxt, l, r
    while True:
        l, r = 2*j+1, 2*j+2
        if l < n and vals[l] > v:
            # Swap with the largest of the two children
            if r < n and vals[r] > vals[l]:
                nxt = r
            else: nxt = l
        elif r < n and vals[r] > v:
            nxt = r
        else:
            break
        indices[j], vals[j] = indices[nxt], vals[nxt]
        j = nxt
    indices[j], vals[j] = i, v


cpdef void insert_heap_2(int[::1] indices, int[::1] vals, int i, int v) nogil:
    ''' Insert (i,v) into the list, which is assumed ordered by vals '''
    # We need to easily be able to identify the largest element in the heap,
    # since that's the one we are kicking out. Thus this is a max-heap with
    # the largest value at 0.
    cdef:
        int n = indices.shape[0]
        int j = 0
        int nxt, l, r
    indices[0], vals[0] = i, v
    while True:
        nxt = j
        l, r = 2*j+1, 2*j+2
        if l < n and vals[l] > vals[nxt]: nxt = l
        if r < n and vals[r] > vals[nxt]: nxt = r
        if nxt == j:
            break
        vals[nxt], vals[j] = vals[j], vals[nxt]
        indices[nxt], indices[j] = indices[j], indices[nxt]
        j = nxt
