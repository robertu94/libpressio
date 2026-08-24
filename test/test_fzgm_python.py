#!/usr/bin/env python3
# test_fzgm_python.py -- FZGpuModules plugin tests via the high-level Python API.
#
# This exists alongside test_fzgm.cc (the C++ gtest suite) specifically to cover
# libpressio's Python option-boxing path, which test_fzgm.cc cannot exercise:
# python_to_pressio_options() (tools/swig/libpressio.py) boxes every Python `int`
# as C++ int64_t and every Python `float` as C++ double, while
# pressio_options::get() requires an *exact* type match with no numeric coercion.
# A plugin option declared as a narrower C++ type (int, float, uint32_t, ...)
# would silently never be set from Python -- this was a real, wide-reaching bug
# in the fzgpumodules plugin (fixed by widening every such field to
# int64_t/double). test_fzgm.cc's C++ pressio_options::set() calls use whatever
# type the test author writes, so they can accidentally match even when Python
# callers would not -- only a real Python-path test catches that class of
# regression.
#
# Run: python3 test_fzgm_python.py <path-to-swig-build-dir>

import sys
pressio_path = sys.argv[1]
sys.path.insert(0, pressio_path)

import numpy as np
import libpressio as lp

FAILED = 0
PASSED = 0


def check(label, condition, detail=""):
    global FAILED, PASSED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        print(f"FAILED: {label} {detail}")


def make_data():
    x = np.linspace(0, 8 * np.pi, 256 * 256).reshape(256, 256)
    return (np.sin(x) + 0.3 * np.cos(3 * x)).astype(np.float32)


data = make_data()

# ── Numeric option readback: every scalar option below is backed by a C++
# field this session widened to int64_t/double specifically so Python's
# int/float boxing would land on it. Setting each from Python and reading it
# straight back (no compress/decompress) isolates the option pipe from
# anything the compression algorithm itself does.

def check_scalar_readback(stage_token, key, value, label=None):
    comp = lp.PressioCompressor.from_config({
        "compressor_id": "fzgpumodules",
        "early_config": {
            "fzgpumodules:stages": [stage_token],
            "fzgpumodules:connections": [],
            key: value,
        },
    })
    got = comp.get_options()[key]
    check(label or key, got == value, f"(set {value!r}, got {got!r})")


check_scalar_readback("quantizer:float:uint16", "fzgpumodules:s0:quant_radius", 999)
check_scalar_readback("quantizer:float:uint16", "fzgpumodules:s0:outlier_capacity", 0.35)
check_scalar_readback("quantizer:float:uint16", "fzgpumodules:s0:dither_seed", 42)
check_scalar_readback("quantizer:float:uint16", "fzgpumodules:s0:dither_strength", 0.5)
check_scalar_readback("rze", "fzgpumodules:s0:chunk_size", 8192)
check_scalar_readback("rze", "fzgpumodules:s0:word_size", 4)
check_scalar_readback("bitpack:uint16", "fzgpumodules:s0:nbits", 8)
check_scalar_readback("bitshuffle", "fzgpumodules:s0:element_width", 2)

comp = lp.PressioCompressor.from_config({"compressor_id": "fzgpumodules"})
comp.set_options({"fzgpumodules:memory_multiplier": 5.0})
comp.set_options({"fzgpumodules:num_streams": 2})
opts = comp.get_options()
check("memory_multiplier", opts["fzgpumodules:memory_multiplier"] == 5.0)
check("num_streams", opts["fzgpumodules:num_streams"] == 2)

# ── End-to-end round-trips exercising the widened options for real, not just
# readback -- confirms the value actually reaches the FZGPUModules engine.

def roundtrip_ok(config, data, eb=1e-3):
    early = dict(config.get("early_config", {}))
    cfg = {"compressor_id": "fzgpumodules", "early_config": early,
           "compressor_config": {"pressio:abs": eb, "pressio:metric": "size",
                                  **config.get("compressor_config", {})}}
    comp = lp.PressioCompressor.from_config(cfg)
    d = np.asarray(comp.decode(comp.encode(data), data.copy()))
    max_err = float(np.abs(data - d).max())
    return max_err <= eb + 1e-6, max_err, comp.get_metrics()["size:compression_ratio"]

ok, max_err, _ = roundtrip_ok({
    "early_config": {
        "fzgpumodules:stages": ["lorenzo:float:uint16", "rze"],
        "fzgpumodules:connections": ["s1 <- s0:codes"],
        "fzgpumodules:s0:quant_radius": 16000,
        "fzgpumodules:s0:outlier_capacity": 0.25,
    },
}, data)
check("lorenzo quant_radius/outlier_capacity round-trip", ok, f"max_err={max_err}")

ok, max_err, ratio = roundtrip_ok({
    "early_config": {
        "fzgpumodules:stages": ["lorenzo:float:uint16", "rze"],
        "fzgpumodules:connections": ["s1 <- s0:codes"],
        "fzgpumodules:s1:chunk_size": 8192,
    },
}, data)
check("rze chunk_size takes effect (ratio differs from default)", ok and ratio != 9.781492537313433,
      f"ratio={ratio}")

# ── New stages from this session: smoke-test that the Python token strings
# work end to end (not exhaustive -- see test_fzgm.cc for full coverage).

for token, dtype, needs_dims in [
    ("szx:float", np.float32, False),
    ("adaptive_bitpack:int32", np.int32, False),
]:
    if dtype == np.int32:
        rng = np.random.default_rng(0)
        d = rng.integers(-1000, 1000, size=256 * 256).astype(np.int32)
        stages = ["tiled_lorenzo:int32", token] if "adaptive_bitpack" in token else [token]
        conns = ["s1 <- s0"] if "adaptive_bitpack" in token else []
        comp = lp.PressioCompressor.from_config({
            "compressor_id": "fzgpumodules",
            "early_config": {"fzgpumodules:stages": stages, "fzgpumodules:connections": conns},
        })
        out = np.asarray(comp.decode(comp.encode(d), d.copy()))
        check(f"{token} lossless round-trip", np.array_equal(d, out))
    else:
        ok, max_err, _ = roundtrip_ok({"early_config": {
            "fzgpumodules:stages": [token], "fzgpumodules:connections": []}}, data)
        check(f"{token} round-trip", ok, f"max_err={max_err}")

# ── Fusion / PREL / centering (new options this session)

comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {"fzgpumodules:fusion": "auto"}})
check("fusion=auto readback", comp.get_options()["fzgpumodules:fusion"] == "auto")

try:
    lp.PressioCompressor.from_config({
        "compressor_id": "fzgpumodules",
        "early_config": {"fzgpumodules:fusion": "bogus"}})
    check("invalid fusion value rejected", False)
except Exception:
    check("invalid fusion value rejected", True)

# PREL's pressio:abs is a *fraction*: abs_eb = pressio:abs * max(|data|).
# roundtrip_ok() checks max_err against the eb it's given, so pass it the
# resolved absolute bound rather than the fraction, matching what PREL
# actually enforces.
prel_frac = 1e-3
prel_abs_bound = prel_frac * float(np.abs(data).max())
comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {
        "fzgpumodules:stages": ["lorenzo:float:uint16", "rze"],
        "fzgpumodules:connections": ["s1 <- s0:codes"],
        "fzgpumodules:error_bound_mode": "prel",
    },
    "compressor_config": {"pressio:abs": prel_frac, "pressio:metric": "size"},
})
d = np.asarray(comp.decode(comp.encode(data), data.copy()))
max_err = float(np.abs(data - d).max())
check("prel error mode round-trip", max_err <= prel_abs_bound + 1e-6, f"max_err={max_err}")

# ── GPU-resident round-trip: compress()/decompress() both accept and can
# return device-resident buffers (anything exposing __cuda_array_interface__
# v3, e.g. CuPy) with no forced device->host copy. Regression coverage for a
# real bug: decompress_typed() used to unconditionally route its output
# through the "malloc" (host) domain regardless of what domain the caller's
# output buffer requested, silently discarding a GPU-resident `out` and
# forcing a D2H copy every decode. Fixed to respect output->domain().
#
# No CuPy/PyTorch dependency here -- fabricate a minimal
# __cuda_array_interface__ wrapper around a raw cudaMalloc'd pointer via
# ctypes + libcudart, so this runs in any CUDA-capable environment.

import ctypes

try:
    cudart = ctypes.CDLL("libcudart.so")
    _HAVE_CUDART = True
except OSError:
    _HAVE_CUDART = False

if _HAVE_CUDART:
    def _cuda_malloc(nbytes):
        ptr = ctypes.c_void_p()
        assert cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)) == 0
        return ptr

    def _cuda_free(ptr):
        cudart.cudaFree(ptr)

    def _cuda_memcpy(dst, src, nbytes, kind):
        assert cudart.cudaMemcpy(dst, src, ctypes.c_size_t(nbytes), ctypes.c_int(kind)) == 0

    class _DeviceArray:
        """Minimal __cuda_array_interface__ v3 wrapper around a cudaMalloc'd pointer."""
        def __init__(self, ptr, shape, dtype):
            self.ptr = ptr
            self.__cuda_array_interface__ = {
                "shape": tuple(shape), "typestr": np.dtype(dtype).str,
                "data": (ptr.value, False), "version": 3, "strides": None,
            }

    h_in = data  # reuse the sin/cos test array from above
    nbytes = h_in.nbytes
    d_in = _cuda_malloc(nbytes)
    _cuda_memcpy(d_in, h_in.ctypes.data_as(ctypes.c_void_p), nbytes, 1)  # H2D
    d_out = _cuda_malloc(nbytes)
    try:
        dev_in = _DeviceArray(d_in, h_in.shape, h_in.dtype)
        dev_out = _DeviceArray(d_out, h_in.shape, h_in.dtype)

        comp = lp.PressioCompressor.from_config({
            "compressor_id": "fzgpumodules",
            "early_config": {
                "fzgpumodules:stages": ["lorenzo:float:uint16", "rze"],
                "fzgpumodules:connections": ["s1 <- s0:codes"],
            },
            "compressor_config": {"pressio:abs": 1e-3, "pressio:metric": "size"},
        })
        compressed = comp.encode(dev_in)
        check("gpu-resident encode() returns a device buffer",
              hasattr(compressed, "__cuda_array_interface__"))

        decompressed = comp.decode(compressed, dev_out)
        check("gpu-resident decode() returns a device buffer",
              hasattr(decompressed, "__cuda_array_interface__"))

        h_check = np.empty_like(h_in)
        out_ptr = ctypes.c_void_p(decompressed.__cuda_array_interface__["data"][0])
        _cuda_memcpy(h_check.ctypes.data_as(ctypes.c_void_p), out_ptr, nbytes, 2)  # D2H
        max_err = float(np.abs(h_in - h_check).max())
        check("gpu-resident round-trip accuracy", max_err <= 1e-3 + 1e-6, f"max_err={max_err}")
    finally:
        _cuda_free(d_in)
        _cuda_free(d_out)
else:
    print("SKIPPED: gpu-resident round-trip (libcudart.so not found)")

print(f"PASSED={PASSED} FAILED={FAILED}")
sys.exit(1 if FAILED else 0)
