# Manylinux Python checks

Run the complete matrix:

```sh
dagger check "dag://?check=libpressio/test-manylinux"
```

The check reads the Python 3.x classifiers from `pyproject.toml` and tests each
advertised CPython version on both manylinux_2_28 x86_64 and aarch64. Each matrix
entry runs sequentially to avoid several native builds exhausting CI resources.
The current images no longer include Python 3.8, so its jobs use the PyPA
`2024.11.30-1` image tag. Free-threaded Python and PyPy are not included.
Cross-architecture jobs require an engine capable of executing that architecture
(e.g. configured QEMU emulation). DNF and ccache caches are scoped by platform
and image, including the dated tag.

To diagnose one entry:

```sh
dagger api call libpressio test-manylinux \
  --version quay.io/pypa/manylinux_2_28_aarch64 \
  --python-version 3.14 --build-jobs 4 pass
```

Each entry:

1. Builds an sdist and wheel with the selected interpreter, then repairs the
   wheel with auditwheel >=6.4.2 in a separate Python 3.12 tooling environment.
   Older auditwheel versions in dated images discard the wheel-root `$ORIGIN`
   search path, breaking imports of the colocated `liblibpressio.so`.
   `std_compat` dependency tests remain disabled.
2. Configures a separate LibPressio build with `BUILD_TESTING=ON` and
   `BUILD_PYTHON_WRAPPER=ON`, builds it, and runs CTest with `--no-tests=error`.
   This includes C++ tests and the Python memory regression suite. The helper
   extension is not included in the production wheel.
3. Installs only the repaired wheel and its declared Python dependencies into a
   new virtual environment in a fresh manylinux container, without the source
   tree or the builder's installed native dependencies. Runs `pip check`, then
   isolated-mode installed-wheel import, noop codec, lifetime, and NumPy
   round-trip tests from `/wheel-tests`.

`--filter-rgx` selects native CTest tests only; it does not skip wheel tests.
The job does not enable optional SZ, CUDA/CuPy, or sanitizer coverage.

After changing the Dagger function signature, refresh the generated schema:

```sh
dagger generate --python-sdk -y
```
