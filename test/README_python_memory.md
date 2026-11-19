# Python memory regression tests on macOS

The suite in `test_python_memory.py` combines exact native allocation/byte
counts, domain-manager branch coverage, repeated ownership replacement, and
NumPy/Python weak-reference lifetime checks. It uses the CORE build and does
not require SZ or CUDA. The selected Python must have NumPy and numcodecs.

## Regular regression check

```sh
cmake -S . -B build-memory \
  -DBUILD_TESTING=ON -DBUILD_PYTHON_WRAPPER=ON \
  -DLIBPRESSIO_BUILD_MODE=CORE
cmake --build build-memory --target pressio pressio_memory_test
ctest --test-dir build-memory -R '^test_python_memory$' --output-on-failure
```

For a longer run, set `LIBPRESSIO_MEMORY_TEST_ITERATIONS=256` when invoking
CTest. The minimum is 64, matching the C++ branch matrix. Copy/move and
high-level tracked-native lifetime tests always repeat at least 1,024 times.

## AddressSanitizer

Use a separate build to catch use-after-free, double-free and buffer overruns:

```sh
cmake -S . -B build-memory-asan \
  -DBUILD_TESTING=ON -DBUILD_PYTHON_WRAPPER=ON \
  -DLIBPRESSIO_BUILD_MODE=CORE \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_SANITIZER=Address
cmake --build build-memory-asan --target pressio pressio_memory_test
ctest --test-dir build-memory-asan -R '^test_python_memory$' --output-on-failure
```

On macOS, CTest automatically injects the ASan runtime located by the selected
C++ compiler into Python using `DYLD_INSERT_LIBRARIES`. No instrumented Python
installation is required. Apple ASan does not provide a general native leak
check: keep the exact counters and the separate `leaks` run below.

## Apple leaks (opt-in)

From the **non-ASan** build:

```sh
cmake --build build-memory --target test_python_memory_leaks
```

This builds the bindings and runs the full suite under
`MallocStackLogging=1 /usr/bin/leaks --atExit`. It is deliberately not a default
CTest because task-port permissions, stack-logger failures, and Python/runtime
shutdown allocations can affect its results. The target propagates failures;
it does not suppress reports or change system security settings.

If `leaks` cannot obtain the Python task port or its stack logger crashes,
that run is inconclusive, not a passing leak check. Diagnose tool/permission
issues separately. Inspect allocation stacks to distinguish libpressio from
Python, NumPy and other imported libraries. Do not add blanket suppressions.

## Investigating reachable memory growth

Counters observe buffers allocated by the test tracking domains, not every
native handle, vector, or malloc-domain allocation. Weakrefs observe retained
Python owners, not arbitrary native objects. `leaks` can detect unreachable
native allocations but may not report memory retained by global references.

For growth with passing tests, use Instruments' **Allocations** template:
launch the selected Python with `test/test_python_memory.py build-memory/tools/swig`
as arguments, record the workload, and compare live allocations between marked
generations. Inspect allocation backtraces for libpressio and its SWIG extension.
Do not use peak RSS alone as a leak assertion: allocator caching is not a leak.
