#!/usr/bin/env python3
"""Deterministic native-buffer lifetime regression tests for the Python API.

Run via CTest (BUILD_TESTING=ON, BUILD_PYTHON_WRAPPER=ON), or pass the
build's tools/swig directory as the first argument. No SZ/CUDA is required.
The tracking domains detect even one leaked buffer, not an RSS threshold.
For native handle/metadata leaks outside those domains, also run this script
under Valgrind/LSan; tracemalloc alone cannot observe C++ allocations.
"""

import gc
import itertools
import os
import sys
import unittest
import weakref
from contextlib import ExitStack, contextmanager
from pathlib import Path

if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    sys.path.insert(0, sys.argv.pop(1))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "swig"))

import libpressio as lp
import numpy as np
import pressio_memory_test as tracking

p = lp.pressio
ITERATIONS = int(os.environ.get("LIBPRESSIO_MEMORY_TEST_ITERATIONS", "64"))
if ITERATIONS < 64:
    raise ValueError("at least 64 iterations are required to match the C++ matrix")
BOOLS = (False, True)
DTYPES = (
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
)


@contextmanager
def handle(value):
    try:
        yield value
    finally:
        p.data_free(value)


class PythonMemoryTests(unittest.TestCase):
    def assert_allocations(self, expected):
        self.assertEqual(tracking.memory_test_allocations(), expected)
        self.assertEqual(tracking.memory_test_bytes(), expected * 16 * 4)

    def setUp(self):
        gc.collect()
        self.assert_allocations(0)

    def tearDown(self):
        gc.collect()
        self.assert_allocations(0)

    def source(self, stack, alternate, view, empty):
        owner = stack.enter_context(handle(tracking.memory_test_new(alternate, empty)))
        if view:
            return stack.enter_context(handle(lp.python_to_new_pressio_data(owner)))
        return owner

    def assert_readable(self, result, source, accessible):
        self.assertEqual(p.data_has_data(result), p.data_has_data(source))
        self.assertEqual(tuple(p.data_dimensions(result)), (16,))
        self.assertEqual(p.data_dtype(result), p.int32_dtype)
        if p.data_has_data(source):
            if accessible:
                self.assertEqual(p.data_ptr(result), p.data_ptr(source))
            else:
                self.assertNotEqual(p.data_ptr(result), p.data_ptr(source))
            np.testing.assert_array_equal(
                lp.pressio_data_to_python(result)[0], np.arange(16, dtype=np.int32)
            )

    def test_exhaustive_domain_manager_branch_matrix_does_not_leak(self):
        # Same dimensions, repeats and state combinations as the C++ matrix.
        # Every result crosses SWIG and is released by the public Python API.
        for iteration, src_alt, dst_alt, view, empty in itertools.product(
            range(ITERATIONS), BOOLS, BOOLS, BOOLS, BOOLS
        ):
            accessible = src_alt == dst_alt
            source_count = int(not empty)
            with self.subTest(
                iteration=iteration,
                source=src_alt,
                destination=dst_alt,
                view=view,
                empty=empty,
            ):
                with ExitStack() as stack:
                    source = self.source(stack, src_alt, view, empty)
                    self.assert_allocations(source_count)
                    for dst_empty in BOOLS:
                        with handle(
                            tracking.memory_test_new(dst_alt, dst_empty)
                        ) as dst:
                            before = tracking.memory_test_allocations()
                            sends = tracking.memory_test_sends()
                            with handle(
                                tracking.memory_test_readable_buffer(dst, source)
                            ) as result:
                                delta = (
                                    (0 if dst_empty else -1)
                                    if accessible or empty
                                    else (1 if dst_empty else 0)
                                )
                                self.assert_allocations(before + delta)
                                self.assert_readable(result, source, accessible)
                                self.assertEqual(
                                    tracking.memory_test_sends() - sends,
                                    int(not accessible and not empty),
                                )
                        self.assert_allocations(source_count)
                    for move_domain in BOOLS:
                        sends = tracking.memory_test_sends()
                        with handle(
                            tracking.memory_test_readable_domain(
                                dst_alt, move_domain, source
                            )
                        ) as result:
                            self.assert_allocations(
                                source_count + int(not accessible and not empty)
                            )
                            self.assert_readable(result, source, accessible)
                            self.assertEqual(
                                tracking.memory_test_sends() - sends,
                                int(not accessible and not empty),
                            )
                        self.assert_allocations(source_count)
                self.assert_allocations(0)
                for move_source in BOOLS:
                    with ExitStack() as stack:
                        source = self.source(stack, src_alt, view, empty)
                        original_ptr = p.data_ptr(source)
                        sends = tracking.memory_test_sends()
                        with handle(
                            tracking.memory_test_writeable(dst_alt, move_source, source)
                        ) as result:
                            reused = accessible and not empty and move_source
                            self.assertTrue(p.data_has_data(result))
                            self.assert_allocations(source_count + int(not reused))
                            self.assertEqual(tracking.memory_test_sends(), sends)
                            if reused:
                                self.assertEqual(p.data_ptr(result), original_ptr)
                    self.assert_allocations(0)

    def test_repeated_copy_move_view_replacement_and_alias_assignment(self):
        with handle(tracking.memory_test_new(False, False)) as current:
            for iteration in range(max(1024, ITERATIONS)):
                with self.subTest(iteration=iteration), ExitStack() as stack:
                    copied = stack.enter_context(handle(p.data_new_clone(current)))
                    self.assert_allocations(2)
                    moved = stack.enter_context(
                        handle(tracking.memory_test_new(False, True))
                    )
                    tracking.memory_test_move_assign(moved, copied)
                    self.assert_allocations(2)
                    view = stack.enter_context(
                        handle(lp.python_to_new_pressio_data(moved))
                    )
                    view_of_view = stack.enter_context(
                        handle(lp.python_to_new_pressio_data(view))
                    )
                    self.assert_allocations(2)
                    for source in (current, moved, view, view_of_view):
                        with handle(tracking.memory_test_new(False, False)) as dst:
                            with handle(
                                tracking.memory_test_readable_buffer(dst, source)
                            ) as readable:
                                self.assert_allocations(2)
                                with handle(
                                    tracking.memory_test_writeable(
                                        False, False, readable
                                    )
                                ):
                                    self.assert_allocations(3)
                            self.assert_allocations(2)
                    # Regression: move-assigning an accessible aliased view
                    # must not lose the allocation owned by the same object.
                    tracking.memory_test_readable_alias_assign(moved)
                    self.assert_allocations(2)
                    np.testing.assert_array_equal(
                        lp.pressio_data_to_python(moved)[0],
                        np.arange(16, dtype=np.int32),
                    )
                    tracking.memory_test_move_assign(current, moved)
                    self.assert_allocations(1)
                self.assert_allocations(1)

    def test_highlevel_raw_encode_and_data_options_release_native_storage(self):
        # Unlike the NumPy-only workload, these high-level paths retain the
        # tracking allocator through native clones and data-valued options.
        for iteration in range(max(1024, ITERATIONS)):
            with self.subTest(iteration=iteration):
                with handle(tracking.memory_test_new(False, False)) as source:
                    compressor = lp.PressioCompressor("noop")
                    with handle(compressor.encode(source)) as encoded:
                        self.assert_allocations(2)
                        np.testing.assert_array_equal(
                            lp.pressio_data_to_python(encoded)[0],
                            np.arange(16, dtype=np.int32),
                        )
                    self.assert_allocations(1)
                    options = lp.python_to_pressio_options({"test:data": source})
                    try:
                        self.assert_allocations(2)
                        converted = lp.pressio_options_to_python(options)
                        np.testing.assert_array_equal(
                            converted["test:data"], np.arange(16, dtype=np.int32)
                        )
                        self.assert_allocations(2)
                    finally:
                        p.options_free(options)
                    self.assert_allocations(1)
                    del compressor, converted
                self.assert_allocations(0)

    def test_numpy_conversions_options_and_compressor_lifetimes(self):
        # All supported dtypes, ranks, zero-length arrays and supplied outputs.
        # Weakrefs additionally detect retained Python/NumPy owners. Use a
        # native leak detector for malloc-domain buffers and SWIG metadata.
        for iteration, dtype, shape in itertools.product(
            range(ITERATIONS), DTYPES, ((16,), (7, 11), (2, 3, 4), (0,), (2, 0, 3))
        ):
            with self.subTest(iteration=iteration, dtype=dtype, shape=shape):
                refs = self.numpy_lifetime_case(dtype, shape)
                gc.collect()
                self.assertTrue(
                    all(ref() is None for ref in refs),
                    "Python API retained an array or compressor",
                )
                self.assert_allocations(0)

    def numpy_lifetime_case(self, dtype, shape):
        data = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
        refs = [weakref.ref(data)]
        with ExitStack() as stack:
            source = stack.enter_context(handle(lp.python_to_new_pressio_data(data)))
            clone = stack.enter_context(handle(p.data_new_clone(source)))
            view = stack.enter_context(handle(lp.python_to_new_pressio_data(source)))
            view_of_view = stack.enter_context(
                handle(lp.python_to_new_pressio_data(view))
            )
            for value in (source, clone, view, view_of_view):
                for supplied in BOOLS:
                    out = np.empty_like(data) if supplied else None
                    result, cleanup = lp.pressio_data_to_python(value, out)
                    self.assertTrue(cleanup)
                    self.assertEqual(result.shape, data.shape)
                    self.assertEqual(result.dtype, data.dtype)
                    np.testing.assert_array_equal(result, data)
                    refs.append(weakref.ref(result))
            options = lp.python_to_pressio_options({"test:data": data})
            try:
                converted = lp.pressio_options_to_python(options)
                np.testing.assert_array_equal(converted["test:data"], data)
                refs.append(weakref.ref(converted["test:data"]))
            finally:
                p.options_free(options)
            compressor = lp.PressioCompressor("noop")
            refs.append(weakref.ref(compressor))
            for supplied in BOOLS:
                encoded = compressor.encode(
                    data, np.empty_like(data) if supplied else None
                )
                decoded = compressor.decode(encoded, np.empty_like(data))
                np.testing.assert_array_equal(decoded, data)
                refs.extend((weakref.ref(encoded), weakref.ref(decoded)))
            encoded_many = compressor.encode_many([data, data])
            decoded_many = compressor.decode_many(
                encoded_many, [np.empty_like(data), np.empty_like(data)]
            )
            for value in decoded_many:
                np.testing.assert_array_equal(value, data)
            refs.extend(weakref.ref(value) for value in encoded_many + decoded_many)
            # Raw-input encode transfers ownership of its result to the caller.
            with handle(compressor.encode(clone)) as encoded_raw:
                with handle(compressor.decode(encoded_raw, source)) as decoded_raw:
                    np.testing.assert_array_equal(
                        lp.pressio_data_to_python(decoded_raw)[0], data
                    )
        return refs


if __name__ == "__main__":
    unittest.main()
