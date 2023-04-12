import numpy as np

# Standard format
#    a0 a1 a2 ... ad
#    b0 b1 b2 ... bd <- 4 bits
# Format as in Quick ADC:
#    a0 b0 c0 ... p0 <- 16'th letter. Whole row is 64 bit
#    a1 b1 c1 ... p1
#    ...
#    ad bd cd ... pd
#    q0 r0 s0 ...    <- New chunk
# Loading data interleaved, since that's how shuffle reads it
#    a0 a1 b0 b1 ... h0 h1
#    i0 i1       ... p0 p1
#    ...
#    i(d-1) id   ... p(d-1) pd
#    q0 q1 r0 ...    <- New chunk
# Also remember that storage inside uint64 is right-to-left


# Table standard format
#    -a1- -a2- ... -a16-
#    -b1- -b2- ... -b16-
#    ...
#    {d}1 ...      {d}16
# Format for Quick ADC:
#    -a1- -a2- ... -a8-
#    -a9- -a10 ... -a16
#    -b1- -b2- ... -b8-
#    -b9- -b10 ... -b16
#    ...


def transform_data(data0):
    """
    Transform data from the standard format to the Quick-ADC format.

    The standard format is a 2D array with n rows and d columns.
    Each entry in the array is a 4-bit value, corresponding to a pointer to a cluster center.

    The Quick ADC format is a transformed version of the standard format, optimized
    for processing in the Quick-ADC algorithm. The data is first reshaped into groups of 16 rows,
    and interleaved by pairs of consecutive columns. Each group of 16 rows becomes a chunk.
    Within a chunk, data is stored in a 64-bit row-major order, with consecutive 4-bit values
    from 16 different rows combined into a single 64-bit value. The storage inside the 64-bit
    value is right-to-left, meaning that the first 4-bit value appears at the least significant
    bits of the 64-bit value.

    The transform_table function does the equivalent transformation for distance tables.
    For the tables, the standard format consists of 8-bit values in a 2D array. In the Quick ADC
    format, the tables are reshaped so that each row is divided into two equal parts and stacked
    one after the other.

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
    assert d % 2 == 0  # Because we load two rows at a time (128bits)
    assert np.all(data0 < 16) and np.all(0 <= data0)
    # Convert data to Quick-ADC format
    data = data0.reshape(n // 16, 16, d)
    data = data.transpose(0, 2, 1)
    # Interleave rows two by two
    data = (                                     # [0,1,2,3,4,5,6,7]
        data.reshape(n // 16, d // 2, 2, 16)     # [0,1,2,3], [4,5,6,7]
        .reshape(n // 16, d // 2, 32, order="F") # [0,4,1,5,2,6,3,7]
        .reshape(n // 16, d, 16)                 # [0,4,1,5], [2,6,3,7]
    )
    # Reversing last dimension for endianess
    data = data[:, :, ::-1]
    # Extra tests for required format
    assert data[0, 0, -1] == data0[0][0]
    assert data[0, 0, -2] == data0[0][1]
    assert data[0, 0, -3] == data0[1][0]
    assert data[0, 0, -4] == data0[1][1]
    assert data[0, 1, -1] == data0[8][0]
    assert data[0, 1, -2] == data0[8][1]
    if d > 2:
        assert data[0, 2, -1] == data0[0][2]
        assert data[0, 2, -2] == data0[0][3]
    assert data.shape == (n // 16, d, 16)
    # Converting last dimension to a single 64 bit number
    # data = np.frompyfunc(lambda x, y: (x << 4 | y), 2, 1).reduce(data, axis=2)
    shifts = np.arange(15, -1, -1, dtype=np.uint64) * 4
    data = (data << shifts).sum(axis=2, dtype=np.uint64)
    data = np.ascontiguousarray(data, dtype=np.uint64)
    assert data.shape == (n // 16, d)
    return data


def transform_tables(tables0):
    d, b = tables0.shape
    assert d % 2 == 0  # Because we load two rows at a time (128bits)
    assert b == 16
    assert tables0.dtype == np.uint8
    # Convert tables to 128 bit format
    tables = tables0.reshape(2 * d, 8)
    # Viewing takes care of swapping the byte order automatically
    tables = tables.view(np.uint64)[:, 0]
    assert tables.shape == (2 * d,)
    return tables


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

    data = data[:, :, ::-1]                               # [0,4,1,5], [2,6,3,7]
    data = data.reshape(chunks, d // 2, 32)               # [0,4,1,5,2,6,3,7]
    data = data.reshape(chunks, d // 2, 2, 16, order='F') # [0,1,2,3], [4,5,6,7]
    data = data.reshape(chunks, d, 16)                    # [0,1,2,3], [4,5,6,7]

    data = data.transpose(0, 2, 1)
    data = data.reshape(n, d)

    return data
