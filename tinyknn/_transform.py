import numpy as np


def transform_data(data0):
    """
    Transform data from the standard format to the Quick-ADC format.

    The standard format is a 2D array with n rows and d columns.
    Each entry in the array is a 4-bit value, corresponding to a pointer to a cluster center.


    Standard format
       a0 a1 a2 ... ad <- Row of 4 * d bits
       b0 b1 b2 ... bd
       ...

    Transpose data as in Quick ADC:
       a0 b0 c0 ... p0 <- 16'th letter. Whole row is 64 bit
       a1 b1 c1 ... p1
       ...
       ad bd cd ... pd
       q0 r0 s0 ...    <- New chunk

    Loading data interleaved, since that's how shuffle reads it
       =====================
       a0 a1 b0 b1 ... h0 h1 <- 64 bits
       i0 i1       ... p0 p1 <- 64 bits
       =====================
       a2 a3 b2 b3 ... h2 h3 <- If using AVX we'll read these 128 bits
       i2 i3       ... p2 p3    together with the previous 128 bits, so we must have d % 4 = 0.
       =====================
       ...
       i(d-1) id   ... p(d-1) pd
       =====================
       q0 q1 r0 ...    <- Beginning of new New chunk
       ...
       =====================

    Also remember that storage inside uint64 is right-to-left

    Parameters
    ----------
    data0 : numpy.ndarray
        A 2D input array with 4-bit entries. Its shape should be (n, d), where
        n is a multiple of 16 and d is even.

    Returns
    -------
    data : numpy.ndarray
        A transformed 2D array with dtype `np.uint64`. Its shape is (n // 16, d)
        since the 4-bit values have been packed in groups of 16.
    """
    n, d = data0.shape
    # Because we load two rows at a time (128bits)
    assert n % 16 == 0, "Number of rows must be divisible by 16"
    # The transform function doesn't care about how many dimensions you load at a time.
    # If you load 2 at a time (SSE) remember to check d%2=0. If you use AVX, check d%4=0.
    #assert avx and d % 4 == 0 or not avx and d % 2 == 0, "Dimensions must be divisible by 4"
    assert np.all(data0 < 16) and np.all(0 <= data0), "Input must be 4 bit values"
    # Split into 16-vector chunks
    data = data0.reshape(n // 16, 16, d)
    # Transpose
    data = data.transpose(0, 2, 1)
    # Interleave rows two by two
    # fmt: off
    data = (                                      # [0,1,2,3,4,5,6,7]
        data.reshape(n // 16, d // 2, 2, 16)      # [0,1,2,3], [4,5,6,7]
        .reshape(n // 16, d // 2, 32, order="F")  # [0,4,1,5,2,6,3,7]
        .reshape(n // 16, d, 16)                  # [0,4,1,5], [2,6,3,7]
    )
    # fmt: on
    # Reversing last dimension for endianess
    data = data[:, :, ::-1]
    # Converting last dimension to a single 64 bit number
    shifts = np.arange(15, -1, -1, dtype=np.uint64) * 4
    data = (data << shifts).sum(axis=2, dtype=np.uint64)
    return np.ascontiguousarray(data, dtype=np.uint64)


def unpack(transformed_data):
    """
    Unpack the transformed data back to its original format.

    Parameters
    ----------
    transformed_data : numpy.ndarray
        A 2D input array with dtype `np.uint64`. It is the output of the transform_data function.

    Returns
    -------
    data : numpy.ndarray
        A 2D array of the original format. Its shape should be (n, d), where
        d is the number of columns in the original data.
    """
    chunks, d = transformed_data.shape
    n = chunks * 16

    shifts = np.arange(15, -1, -1, dtype=np.uint64) * 4
    data = (transformed_data[..., np.newaxis] >> shifts) & 0xF

    # fmt: off
    data = data[:, :, ::-1]                                # [0,4,1,5], [2,6,3,7]
    data = data.reshape(chunks, d // 2, 32)                # [0,4,1,5,2,6,3,7]
    data = data.reshape(chunks, d // 2, 2, 16, order="F")  # [0,1,2,3], [4,5,6,7]
    data = data.reshape(chunks, d, 16)                     # [0,1,2,3], [4,5,6,7]
    # fmt: on

    data = data.transpose(0, 2, 1)
    data = data.reshape(n, d)

    return data


def transform_tables(tables0):
    """
    Table standard format
       -a1- -a2- ... -a16-
       -b1- -b2- ... -b16-
       ...
       {d}1 ...      {d}16

    Format for Quick ADC:
       -a1- -a2- ... -a8-
       -a9- -a10 ... -a16
       -b1- -b2- ... -b8-
       -b9- -b10 ... -b16
       ...
    """
    d, b = tables0.shape
    #assert d % 2 == 0  # Because we load two rows at a time (128bits)
    assert b == 16
    assert tables0.dtype == np.uint8
    # Convert tables to 128 bit format
    tables = tables0.reshape(2 * d, 8)
    # Viewing takes care of swapping the byte order automatically
    tables = tables.view(np.uint64)[:, 0]
    assert tables.shape == (2 * d,)
    return tables
