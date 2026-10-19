#!/usr/bin/env python3
"""FZGM configuration transactions must preserve values and live pipelines on rejection."""
import sys
import os
import subprocess
import unittest
binding_dir = sys.argv.pop(1)
sys.path.insert(0, binding_dir)
import libpressio as lp
import libpressio_fzgpumodules as fzgm
import numpy as np


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.comp = (fzgm.Chain()
            .add("quantizer:float:uint32", linear_mode=True)
            .add("lorenzo:int32", from_port="codes", block_size=32)
            .add("adaptive_bitpack:int32", block_size=32)
            .configure(eb=1e-3, memory_strategy="preallocate", specialization="off"))
        self.x = np.sin(np.arange(131, dtype=np.float32))
        self.baseline = self.comp.encode(self.x)
        self.before = self.comp.get_options()

    def assert_rollback(self, options):
        with self.assertRaises(lp.PressioException): self.comp.set_options(options)
        after = self.comp.get_options()
        for key in self.before:
            if key.startswith("fzgpumodules:") or key == "pressio:abs":
                np.testing.assert_equal(after[key], self.before[key], err_msg=key)
        np.testing.assert_array_equal(self.comp.encode(self.x), self.baseline)
        self.assertFalse(self.comp.get_metrics()["fzgpumodules:rebuilt"])

    def test_rollback_after_top_level_failure(self):
        for key, value in (("memory_strategy", "bogus"), ("error_bound_mode", "bogus"),
                           ("specialization", "bogus"), ("fusion", "auto")):
            with self.subTest(key=key):
                self.assert_rollback({"pressio:abs": 0.25,
                    "fzgpumodules:stages": ["bitshuffle"],
                    "fzgpumodules:connections": [], "fzgpumodules:"+key: value})

    def test_graph_capture_option_is_removed(self):
        self.assertNotIn("fzgpumodules:graph_mode", self.comp.get_options())
        self.assertNotIn("fzgpumodules:graph_mode", self.comp.get_documentation())
        for value in (True, False, "true"):
            with self.subTest(value=value):
                self.assert_rollback({"pressio:abs": 0.25, "fzgpumodules:graph_mode": value})
        with self.assertRaisesRegex(lp.PressioException, "graph_mode was removed"):
            fzgm.Chain().add("bitshuffle").configure(graph_mode=True)

    def test_rollback_after_earlier_stage_parameter_changed(self):
        self.assert_rollback({"pressio:abs": 0.25,
                              "fzgpumodules:s0:dither_seed": 42,
                              "fzgpumodules:s1:block_size": "wrong type"})

    def test_top_level_wrong_types_and_numeric_domains(self):
        for key, value in (("num_streams", "two"), ("num_streams", 0),
                           ("memory_multiplier", float("nan")), ("memory_multiplier", -1.0), ("memory_multiplier", 1e300),
                           ("dims", np.array([-1, 1, 1], dtype=np.int64)),
                           ("dims", np.array([1.5, 1, 1], dtype=np.float64)),
                           ("dims", "invalid")):
            with self.subTest(key=key):
                self.assert_rollback({"pressio:abs": 0.25, "fzgpumodules:"+key: value})
        for bound in (-1.0, float("inf")):
            with self.subTest(bound=bound): self.assert_rollback({"pressio:abs": bound})

    def test_successful_update_rebuilds_once_and_noop_preserves_pipeline(self):
        self.comp.set_options({"pressio:abs": 5e-4, "fzgpumodules:s0:dither_seed": 42})
        self.assertEqual(self.comp.get_options()["fzgpumodules:s0:dither_seed"], 42)
        self.comp.encode(self.x)
        self.assertTrue(self.comp.get_metrics()["fzgpumodules:rebuilt"])
        self.comp.set_options({"pressio:abs": 5e-4, "fzgpumodules:s0:dither_seed": 42})
        self.comp.encode(self.x)
        self.assertFalse(self.comp.get_metrics()["fzgpumodules:rebuilt"])

    def test_inactive_stage_settings_survive_transactions(self):
        self.comp.set_options({"fzgpumodules:s0:dither_seed": 42})
        self.comp.set_options({"fzgpumodules:stages": ["bitshuffle"], "fzgpumodules:connections": []})
        with self.assertRaises(lp.PressioException):
            self.comp.set_options({"pressio:abs": 0.25, "fzgpumodules:memory_strategy": "bogus"})
        self.comp.set_options({"fzgpumodules:stages": self.before["fzgpumodules:stages"],
                               "fzgpumodules:connections": self.before["fzgpumodules:connections"]})
        self.assertEqual(self.comp.get_options()["fzgpumodules:s0:dither_seed"], 42)

    def test_rejection_preserves_pending_rebuild_and_decode_state(self):
        self.comp.set_options({"pressio:abs": 5e-4})
        with self.assertRaises(lp.PressioException):
            self.comp.set_options({"pressio:abs": 0.25, "fzgpumodules:memory_strategy": "bogus"})
        self.assertEqual(self.comp.get_options()["pressio:abs"], 5e-4)
        encoded = self.comp.encode(self.x)
        self.assertTrue(self.comp.get_metrics()["fzgpumodules:rebuilt"])
        self.comp.decode(encoded)
        with self.assertRaises(lp.PressioException):
            self.comp.set_options({"pressio:abs": 0.25, "fzgpumodules:memory_strategy": "bogus"})
        self.assertEqual(self.comp.get_options()["pressio:abs"], 5e-4)
        self.comp.encode(self.x)
        self.assertTrue(self.comp.get_metrics()["fzgpumodules:rebuilt"])

    def test_failed_pipeline_build_can_retry_and_recover(self):
        self.comp.set_options({"fzgpumodules:connections": ["s1 <- s0:missing_port"]})
        for _ in range(2):
            with self.assertRaises(lp.PressioException):
                self.comp.encode(self.x)
        self.comp.set_options({"fzgpumodules:connections": self.before["fzgpumodules:connections"]})
        encoded = self.comp.encode(self.x)
        self.assertTrue(self.comp.get_metrics()["fzgpumodules:rebuilt"])
        restored = self.comp.decode(encoded)
        np.testing.assert_allclose(restored, self.x, atol=1e-3, rtol=0)
        # Archives returned before the failed builds retain independent ownership.
        np.testing.assert_allclose(self.comp.decode(self.baseline), self.x, atol=1e-3, rtol=0)


class ExplicitPipelineTests(unittest.TestCase):
    def test_encoding_requires_pipeline_and_can_recover(self):
        comp = lp.PressioCompressor.from_config({"compressor_id": "fzgpumodules"})
        options = comp.get_options()
        self.assertEqual(options["fzgpumodules:stages"], [])
        self.assertEqual(options["fzgpumodules:connections"], [])
        self.assertEqual(options["pressio:abs"], 1e-3)
        x = np.sin(np.arange(131, dtype=np.float32))
        with self.assertRaisesRegex(lp.PressioException, "explicit pipeline"):
            comp.encode(x)
        comp.set_options({"fzgpumodules:stages": ["lorenzo:float:uint16"]})
        encoded = comp.encode(x)
        comp.set_options({"fzgpumodules:stages": [], "fzgpumodules:connections": []})
        with self.assertRaisesRegex(lp.PressioException, "explicit pipeline"):
            comp.encode(x)
        # Neither creating nor clearing an encoder configuration prevents archive decoding.
        for dec in (comp, fzgm.Compressor.from_config({"compressor_id": "fzgpumodules"})):
            np.testing.assert_allclose(dec.decode(encoded, np.empty_like(x)), x, atol=1e-3, rtol=0)

    def test_empty_builder_requires_a_stage(self):
        comp = fzgm.Chain().configure()
        with self.assertRaisesRegex(lp.PressioException, "explicit pipeline"):
            comp.encode(np.ones(32, dtype=np.float32))


class RuntimeFailureTests(unittest.TestCase):
    def test_no_gpu_reports_stream_error_without_breaking_configuration(self):
        # Use a fresh process because CUDA caches device visibility after initialization.
        code = """
import sys
sys.path.insert(0, sys.argv[1])
import libpressio as lp
import libpressio_fzgpumodules as fzgm
import numpy as np
comp = fzgm.Chain().add("bitshuffle").configure()
assert comp.get_options()["fzgpumodules:specialization"] == "off"
for _ in range(2):
    try:
        comp.encode(np.ones(32, dtype=np.float32))
    except lp.PressioException as error:
        assert "CUDA stream creation failed" in str(error), str(error)
    else:
        raise AssertionError("Encoding succeeded without a visible GPU")
"""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        result = subprocess.run([sys.executable, "-c", code, binding_dir],
                                env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__": unittest.main()
