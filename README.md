# LibPressio

[![Build Status](https://travis-ci.org/robertu94/libpressio.svg?branch=master)](https://travis-ci.org/robertu94/libpressio)


*the upstream version of this code is found at [at the CODARCode organization](https://github.com/CODARcode/libpressio)*

Pressio is latin for compression.  LibPressio is a C++ library with C compatible bindings to abstract between different lossless and lossy compressors and their configurations.  It solves the problem of having to having to write separate application level code for each lossy compressor that is developed.  Instead, users write application level code using LibPressio, and the library will make the correct underlying calls to the compressors.  It provides interfaces to represent data, compressors settings, and compressors.

Documentation for the `master` branch can be [found here](https://robertu94.github.io/libpressio/)

## Using LibPressio

Here is a minimal example with error checking of how to use LibPressio:

```c
#include <libpressio.h>
#include <sz.h>

// provides input function, found in ./test
#include "make_input_data.h"

int
main(int argc, char* argv[])
{
  // get a handle to a compressor
  struct pressio* library = pressio_instance();
  struct pressio_compressor* compressor = pressio_get_compressor(library, "sz");

  // configure metrics
  const char* metrics[] = { "size" };
  struct pressio_metrics* metrics_plugin =
    pressio_new_metrics(library, metrics, 1);
  pressio_compressor_set_metrics(compressor, metrics_plugin);

  // configure the compressor
  struct pressio_options* sz_options =
    pressio_compressor_get_options(compressor);

  pressio_options_set_integer(sz_options, "sz:error_bound_mode", ABS);
  pressio_options_set_double(sz_options, "sz:abs_err_bound", 0.5);
  if (pressio_compressor_check_options(compressor, sz_options)) {
    printf("%s\n", pressio_compressor_error_msg(compressor));
    exit(pressio_compressor_error_code(compressor));
  }
  if (pressio_compressor_set_options(compressor, sz_options)) {
    printf("%s\n", pressio_compressor_error_msg(compressor));
    exit(pressio_compressor_error_code(compressor));
  }

  // load a 300x300x300 dataset into data created with malloc
  double* rawinput_data = make_input_data();
  size_t dims[] = { 300, 300, 300 };
  struct pressio_data* input_data =
    pressio_data_new_move(pressio_double_dtype, rawinput_data, 3, dims,
                          pressio_data_libc_free_fn, NULL);

  // creates an output dataset pointer
  struct pressio_data* compressed_data =
    pressio_data_new_empty(pressio_byte_dtype, 0, NULL);

  // configure the decompressed output area
  struct pressio_data* decompressed_data =
    pressio_data_new_empty(pressio_double_dtype, 3, dims);

  // compress the data
  if (pressio_compressor_compress(compressor, input_data, compressed_data)) {
    printf("%s\n", pressio_compressor_error_msg(compressor));
    exit(pressio_compressor_error_code(compressor));
  }

  // decompress the data
  if (pressio_compressor_decompress(compressor, compressed_data,
                                    decompressed_data)) {
    printf("%s\n", pressio_compressor_error_msg(compressor));
    exit(pressio_compressor_error_code(compressor));
  }

  // get the compression ratio
  struct pressio_options* metric_results =
    pressio_compressor_get_metrics_results(compressor);
  double compression_ratio = 0;
  if (pressio_options_get_double(metric_results, "size:compression_ratio",
                                 &compression_ratio)) {
    printf("failed to get compression ratio\n");
    exit(1);
  }
  printf("compression ratio: %lf\n", compression_ratio);

  // free the input, decompressed, and compressed data
  pressio_data_free(decompressed_data);
  pressio_data_free(compressed_data);
  pressio_data_free(input_data);

  // free options and the library
  pressio_options_free(sz_options);
  pressio_options_free(metric_results);
  pressio_compressor_release(compressor);
  pressio_release(library);
  return 0;
}
```

## Getting Started

After skimming the example, LibPressio has 6 major headers that you will need to use:

Type                  | Use 
----------------------|-------------------
`pressio.h`             | Error reporting and aquiring handles to compressors
`pressio_compressor.h`  | Used to compress and decompress data, provided by plugins
`pressio_data.h`        | Represents data and associated metadata (size, type, dimentionality, memory ownership)
`pressio_options.h`     | Maps between names and values, used for options for compressors and metrics results
`pressio_metrics.h`     | A set of metrics to run while compressors run
`pressio_io.h`     | An extension header that provides methods to load or store data from/to persistent storage

All of these are included by the convience header `libpressio.h`.

You can pick up the more advanced features as you need them.


You can also find more examples in `test/`

## Supported Compressors and Metrics

Libpressio provides a number of builtin compressor and metrics modules.
All of these are disabled by default.
They can be enabled by passing the corresponding `LIBPRESSIO_HAS_*` variable to CMake.

Additionally, Libpressio is extensible.
For information on writing a compressor plugin see [Writing a Compressor Plugin](@ref writingacompressor)
For information on writing a metrics plugin see [Writing a Metrics Plugin](@ref writingametric)


### Compressor Plugins

See the [compressor settings page](@ref pressiooptions) for information on how to configure them.

+ `sz` -- the SZ error bounded lossy compressor
+ `zfp` -- the ZFP error bounded lossy compressor
+ `mgard` -- the MGARD error bounded lossy compressor
+ `blosc` -- the blosc lossless compressor
+ `magick` -- the ImageMagick image compression/decompression library
+ `fpzip` -- the fpzip floating point lossless compressor
+ `noop` -- a dummy compressor useful performance evaluation, testing, and introspection
+ `sampling` -- a compressor which does naive, with out replacement, and with replacement sampling
+ `transpose` -- a meta-compressor which performs a transpose.
+ `resize` -- a meta-compressor which preforms a reshape operation.

### Metrics Plugins

See the [metrics results page](@ref metricsresults) for information on what they produce

+ `time` -- time information on each compressor API
+ `error_stat` -- statistics on the difference between the uncompressed and decompressed values that can be computed in one pass in linear time.
+ `spatial_error` -- computes relative spatial error
+ `pearson` -- computes the pearson coefficient of correlation and pearson coefficient of determination.
+ `size` -- information on the size of the compressed and decompressed data
+ `external` -- run an external program to collect some metrics, see [using an external metric for more information](@ref usingexternalmetric)

## Dependencies

Libpressio unconditionally requires:

+ `cmake` version `3.13` or later (3.14 required for python bindings)
+ `pkg-config` version `1.6.3` or later
+ either:
  + `gcc-4.8.5` or later
  + `clang-7.0.0` or later using either `libc++` or `libstdc++`.  Beware that system libraries may need to be recompiled with `libc++` if using `libc++`

Libpressio additionally optionally requires:

+ `Doxygen` version 1.8.15 or later to generate documentation
+ `HDF5` version 1.10.0 or later for HDF5 data support
+ `ImageMagick` version 6.9.7 or later for ImageMagick image support.  Version 7 or later supports additional data types.
+ `blosc` version 1.14.2 for lossless compressor support via blosc
+ `boost` version 1.53 to compile on a c++14 or earlier compiler
+ `fpzip` version 1.3 for fpzip support
+ `numpy` version `1.14.5` or later and its dependencies to provide the python bindings
+ `swig` version 3.0.12 or later for python support
+ `sz` commit `7b7463411f02be4700d13aac6737a6a9662806b4` or later and its dependencies to provide the SZ plugin
+ `zfp` commit `e8edaced12f139ddf16167987ded15e5da1b98da` or later and its dependencies to provide the ZFP plugin
+ `python` 3.4 or later for the python bindings
+ `lua` or `luajit` version 5.1 or later to provide custom composite metrics.  NOTE compiling with Lua support requires c++17 or later (i.e. gcc 7 or later, and clang 3.9 or later; see Sol2 for current requirements).
+ `sol2` version 3.2.0 or later to provide custom composite metrics
+ `OpenMP` development libraries and headers for your compiler compatible with OpenMP Standard 3 or later to accelerate computation of some metrics.
+ `MPI` development libraries and headers supporting MPI-2 (specifically MPI\_Comm\_spawn using the `workdir` info option) to provide the external metrics `mpispawn` launch method
+ `PETSc` version 3.12.1 or later to provide PETSc binary format IO support

It is also possible to build and run libpressio via Docker using the docker files in the `docker` directory.  This functionality should be considered deprecated and will be removed in a later release, please you spack instead.

## Installing LibPressio using Spack

LibPressio can be built using [spack](https://github.com/spack/spack/).

```bash
git clone https://github.com/robertu94/spack_packages robertu94_packages
spack repo add robertu94_packages
spack install libpressio
```

You can substantially reduce install times by not installing ImageMagick and PETSc support.

```
spack install libpressio~magick~petsc
```

## Configuring LibPressio Manually

LibPressio uses cmake to configure build options.  See CMake documentation to see how to configure options

+ `CMAKE_INSTALL_PREFIX` - install the library to a local directory prefix
+ `BUILD_DOCS` - build the project documentation
+ `BUILD_TESTING` - build the test cases

## Building and Installing LibPressio

To build and tests and install the library only.

```bash
BUILD_DIR=build
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake ..
make
make test
make install
```

To build the documentation:


```bash
BUILD_DIR=build
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake .. -DBUILD_DOCS=ON
make docs
# the html docs can be found in $BUILD_DIR/html/index.html
# the man pages can be found in $BUILD_DIR/man/
```

To build on a C++11 compiler: (make sure boost is available)

```
BUILD_DIR=build
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake -DLIBPRESSIO_CXX_VERSION=11 ..
make
```

To build the experimental python bindings:

```
BUILD_DIR=build
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake .. -DBUILD_PYTHON_WRAPPER=ON
make
make install
```

To disable building the test cases

```
BUILD_DIR=build
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake .. -DBUILD_TESTING=OFF
make
ctest .
```


## Option Names

LibPressio uses a key-value system to refer to configuration settings.

Each compressor may find specific configuration settings for its specific compressor with settings beginning with its compressor id as prefix (i.e. configurations for SZ begin with `sz:`).  [Refer to the specific compressors documentation](@ref pressiooptions) for further documentation for each settings.

The prefixes `metrics:` and `pressio:` are reserved for future use.

## Stability

As of version 1.0.0, LibPressio will follow the following API stability guidelines:

+ The functions defined in files in `./include` excluding files in the `./include/libpressio_ext/` or its subdirectories may be considered to be stable.  Furthermore, all files in this set are C compatible.
+ The functions defined in files in `./include/libpressio_ext/` are to be considered unstable.
+ The functions and modules defined in the python bindings are unstable.

Stable means:

+ New APIs may be introduced with the increase of the minor version number.
+ APIs may gain additional overloads for C++ compatible interfaces with an increase in the minor version number.
+ An API may change the number or type of parameters with an increase in the major version number.
+ An API may be removed with the change of the major version number

Unstable means:

+ The API may change for any reason with the increase of the minor version number

Additionally, the performance of functions, memory usage patterns may change for both stable and unstable code with the increase of the patch version.

## Bug Reports

Please files bugs to the Github Issues page on the CODARCode libpressio repository.

Please read this post on [how to file a good bug report](https://codingnest.com/how-to-file-a-good-bug-report/).  After reading this post, please provide the following information specific to libpressio:

+ Your OS version and distribution information, usually this can be found in `/etc/os-release`
+ the output of `cmake -L $BUILD_DIR`
+ the version of each of libpressio's dependencies listed in the README that you have installed. Where possible, please provide the commit hashes.

