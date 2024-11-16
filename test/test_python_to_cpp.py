#!/usr/bin/env python
import libpressio as lp
import numpy as np
import sys


data = np.random.rand(50,50)
lp_data = lp.python_to_pressio_data(data)
data2 = lp.pressio_data_to_python(lp_data)[0]
assert (data == data2).all(), "converting from python to C++ and back should not change data"

try:
    import cupy as cp
    # tests for cupy conversion
    cu_data1 = cp.random.rand(50,50)
    cu_lp_data = lp.python_to_pressio_data(cu_data1)
    cu_data2 = cp.asarray(lp.pressio_data_to_python(cu_lp_data)[0])
    assert (cu_data1 == cu_data2).all(), "converting from cupy python to C++ and back should not change data"
except ImportError:
    print("cupy not available; tests skipped", file=sys.stderr)
