#!/usr/bin/env python3
"""Exercise Python bindings without requiring an external compressor."""
import sys
import unittest

sys.path[:0] = sys.argv[1:3]
del sys.argv[1:3]

import libpressio as lp
import numpy as np
from numcodecs.abc import Codec


class PythonCompatibilityTests(unittest.TestCase):
    def test_python3_options(self):
        values = {"integer": 7, "real": 1.25, "text": "hello", "boolean": True}
        options = lp.python_to_pressio_options(values)
        try:
            self.assertEqual(lp.pressio_options_to_python(options), values)
        finally:
            lp.pressio.options_free(options)

    def test_numpy_output_survives_native_release(self):
        array = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        data = lp.pressio.io_data_from_numpy(array)
        try:
            restored = lp.pressio.io_data_to_numpy(data)
        finally:
            lp.pressio.data_free(data)
        np.testing.assert_array_equal(restored, array)
        restored[0, 0, 0] = -1
        self.assertEqual(array[0, 0, 0], 0)

    def test_noop_numcodecs_roundtrip(self):
        compressor = lp.PressioCompressor.from_config({"compressor_id": "noop"})
        self.assertIsInstance(compressor, Codec)
        for shape in ((17,), (3, 5), (2, 3, 4), (2, 3, 4, 5)):
            for dtype in (np.float32, np.float64, np.int32, np.uint16):
                with self.subTest(shape=shape, dtype=dtype):
                    array = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
                    restored = compressor.decode(compressor.encode(array), np.empty_like(array))
                    np.testing.assert_array_equal(restored, array)
                    self.assertEqual(restored.dtype, array.dtype)
                    self.assertEqual(restored.shape, array.shape)

    def test_noop_configuration_roundtrip(self):
        compressor = lp.PressioCompressor.from_config({"compressor_id": "noop"})
        restored = lp.PressioCompressor.from_config(compressor.get_config())
        array = np.arange(31, dtype=np.float32)
        np.testing.assert_array_equal(
            restored.decode(restored.encode(array), np.empty_like(array)), array)


if __name__ == "__main__":
    unittest.main()
