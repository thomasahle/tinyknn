#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

cdef extern from *:
    ctypedef int uint64_t "__uint64_t"
    ctypedef int uint128_t "__uint128_t"

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


cdef union u128:
    __m128i simd
    char[16] byts

# We can use pshufb to do 16 1byte lookups in one operation.
# See https://arxiv.org/pdf/1812.09162.pdf
cpdef void query_pq_sse(uint64_t[:,::1] data, uint64_t[::1] tables, uint64_t[::1] out) nogil:
    ''' Given a N x D dataset quantized into byte sizes chunks,
        looks up each value in a table out outputs into `out`. '''
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
    cdef:
        int n = data.shape[0]
        int block_size = data.shape[1]//2 # We read two 64bit chunks at a time
        int i, j
        __m128i hi_table, lo_table, block_dists
        __m128i block, block_masked, lo_block, hi_block, dists
        uint64_t low_mask64 = 0x0f0f0f0f0f0f0f0f
        __m128i low_mask = _mm_set_epi64x(low_mask64, low_mask64)

    for i in range(n):
        # Since distances are positive, but we add as signed values, maybe we
        # could get some extra precision by initializaing this to -128 in each lane?
        #block_dists = _mm_set_epi64x(0, 0)
        block_dists = _mm_setzero_si128()
        for j in range(block_size):
            # Do we need to use _mm_loadu_si128 or _mm_loadu_si128 to load data?
            # I think they are only needed when data can be unaligned, but if we
            # use a numpy uint128, won't it be aligned?
            block = _mm_loadu_si128(<__m128i*> &data[i][2*j])
            # Low comps
            lo_table = _mm_loadu_si128(<__m128i*> &tables[4*j])
            block_masked = _mm_and_si128(block, low_mask);
            dists = _mm_shuffle_epi8(lo_table, block_masked)
            #dists = _mm_shuffle_epu8(lo_table, block & low_mask)
            # Use epi instead of epu for signed addition
            block_dists = _mm_adds_epu8(block_dists, dists)
            # High comps
            hi_table = _mm_loadu_si128(<__m128i*> &tables[4*j+2])
            block_masked = _mm_srli_epi64(block, 4) # >> 4
            block_masked = _mm_and_si128(block_masked, low_mask) # & low_mask
            dists = _mm_shuffle_epi8(hi_table, block_masked)
            #dists = _mm_shuffle_epu8(hi_table, (block >> 4) & low_mask)
            block_dists = _mm_adds_epu8(block_dists, dists)
        # caller side?
        #out[2*i] = <uint128_t>block_dists
        out[2*i]   = _mm_extract_epi64(block_dists, 0)
        out[2*i+1] = _mm_extract_epi64(block_dists, 1)



