#!/usr/bin/env python
import libpressio as lp
import numpy as np
import sys


for shape in [(50, 50), (7, 11), (2, 3, 4)]:
    for dtype in [np.float64, np.float32, np.int32, np.uint8, np.bool_]:
        data = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
        lp_data = lp.python_to_new_pressio_data(data)
        try:
            data2 = lp.pressio_data_to_python(lp_data)[0]
            assert data2.shape == data.shape, (
                "round-trip conversion should preserve shape"
            )
            assert data2.dtype == data.dtype, (
                "round-trip conversion should preserve dtype"
            )
            np.testing.assert_array_equal(data, data2)
        finally:
            lp.pressio.data_free(lp_data)

try:
    import cupy as cp

    # tests for cupy conversion
    cu_data1 = cp.random.rand(50, 50)
    cu_lp_data = lp.python_to_new_pressio_data(cu_data1)
    cu_data2 = cp.asarray(lp.pressio_data_to_python(cu_lp_data)[0])
    assert (cu_data1 == cu_data2).all(), (
        "converting from cupy python to C++ and back should not change data"
    )
except ImportError:
    print("cupy not available; tests skipped", file=sys.stderr)
