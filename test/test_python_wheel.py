#!/usr/bin/env python3
"""Tests for the installed, repaired wheel; no build-tree imports or plugins."""

import gc
import sys
import unittest
from pathlib import Path

import libpressio as lp
import numpy as np
import pressio
import _pressio


class WheelTests(unittest.TestCase):
    def test_imports_are_from_virtualenv(self):
        for module in (lp, pressio, _pressio):
            # relative_to also works on the oldest supported Python (3.8).
            Path(module.__file__).resolve().relative_to(Path(sys.prefix).resolve())
        self.assertIn("noop", lp.supported_compressors())

    def test_noop_round_trip(self):
        for dtype in (
            np.float32,
            np.float64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.bool_,
        ):
            for shape in ((7, 11), (2, 3, 4)):
                with self.subTest(dtype=dtype, shape=shape):
                    source = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
                    codec = lp.PressioCompressor("noop")
                    encoded = codec.encode(source)
                    decoded = codec.decode(encoded, np.empty_like(source))
                    self.assertEqual(decoded.shape, source.shape)
                    self.assertEqual(decoded.dtype, source.dtype)
                    np.testing.assert_array_equal(decoded, source)
                    del codec
                    gc.collect()
                    np.testing.assert_array_equal(decoded, source)

    def test_options_and_configuration(self):
        codec = lp.PressioCompressor("noop", name="wheel-test")
        self.assertEqual(codec.get_name(), "wheel-test")
        self.assertIsInstance(codec.get_options(), dict)
        self.assertIsInstance(codec.get_configuration(), dict)
        self.assertIsInstance(codec.get_documentation(), dict)
        codec.set_options({})
        restored = lp.PressioCompressor.from_config(codec.get_config())
        self.assertEqual(restored.get_name(), "wheel-test")
        self.assertIsInstance(codec.get_metrics(), dict)

    def test_repeated_lifetimes(self):
        source = np.arange(32, dtype=np.float64)
        for _ in range(128):
            codec = lp.PressioCompressor("noop")
            np.testing.assert_array_equal(
                codec.decode(codec.encode(source), np.empty_like(source)), source
            )
        gc.collect()


if __name__ == "__main__":
    unittest.main(verbosity=2)
