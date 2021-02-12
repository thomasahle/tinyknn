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
    # Takes data and tables in standard format and convers to quick-adc format
    # Entries in data0 should be 4 bits
    # Entries in tables0 should be 8 bits
    n, d = data0.shape
    assert d % 2 == 0 # Because we load two rows at a time (128bits)
    assert np.all(data0 < 16) and np.all(0 <= data0)
    # Convert data to Quick-ADC format
    data = data0.reshape(n//16, 16, d)
    data = data.transpose(0, 2, 1)
    # Interleave rows two by two
    data = data.reshape(n//16, d//2, 2, 16) \
                 .reshape(n//16, d//2, 32, order='F') \
                 .reshape(n//16, d, 16)
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
    assert data.shape == (n//16, d, 16)
    # Converting last dimension to a single 64 bit number
    # TODO: This is quite slow. Maybe there's a fast way like the view(np.unit8)
    #       we use in transform_query?
    data = np.frompyfunc(lambda x,y: (x<<4 | y), 2, 1).reduce(data, axis=2)
    data = np.ascontiguousarray(data, dtype=np.uint64)
    assert data.shape == (n//16, d)
    return data

def transform_tables(tables0):
    d, b = tables0.shape
    assert d % 2 == 0 # Because we load two rows at a time (128bits)
    assert b == 16
    assert tables0.dtype == np.uint8
    #assert np.all(-128 <= tables0) and np.all(tables0 < 128)
    assert np.all(0 <= tables0) and np.all(tables0 < 256)
    # Convert tables to 128 bit format
    tables = tables0.reshape(2*d, 8)
    # Viewing takes care of swapping the byte order automatically
    tables = tables.view(np.uint64)[:,0]
    assert tables.shape == (2*d,)
    return tables

