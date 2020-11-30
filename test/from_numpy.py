#!/usr/bin/env python
import sys
pressio_path = sys.argv[1]
sys.path.insert(0, pressio_path)
import pressio
import numpy as np

SIZES = [(3,), (3,5), (3,5,7), (3,5,7,11)]
FLOAT_DTYPES = [np.float32, np.float64]
INT_DTYPES = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
rng = np.random.default_rng()
FAILED = 0
PASSED = 0

def test_float(dtype, size):
    convert_back_and_forth(rng.random(size=size, dtype=dtype))

def test_integer(dtype, size):
    convert_back_and_forth(rng.integers(1, 20, size=size, dtype=dtype))

def convert_back_and_forth(nd_array):
    try:
        p_data = pressio.io_data_from_numpy(nd_array)
        n_data = pressio.io_data_to_numpy(p_data)
        pressio.data_free(p_data)
        assert np.array_equal(nd_array, n_data), f"FAILED nd_array size={nd_array.shape}, dtype={nd_array.dtype}"
    except (AssertionError,TypeError) as e:
        print(e)
        global FAILED
        FAILED += 1
    else:
        global PASSED
        PASSED += 1



for size in SIZES:
    for float_dtype in FLOAT_DTYPES:
        test_float(float_dtype, size)
    for int_dtype in INT_DTYPES:
        test_integer(int_dtype, size)
print("PASSED=", PASSED, "FAILED=", FAILED)
sys.exit(FAILED != 0)
