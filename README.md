# LibPressio

[![Build Status](https://travis-ci.org/robertu94/libpressio.svg?branch=master)](https://travis-ci.org/robertu94/libpressio)

*the upstream version of this code is found at [at the CODARCode organization](https://github.com/CODARcode/libpressio)*

Pressio is latin for compression.  LibPressio is a C++ library with C compatible bindings to abstract between different lossless and lossy compressors and their configurations.  It solves the problem of having to having to write separate application level code for each lossy compressor that is developed.  Instead, users write application level code using LibPressio, and the library will make the correct underlying calls to the compressors.  It provides interfaces to represent data, compressors settings, and compressors.


## Configuring LibPressio

LibPressio uses cmake to configure build options.  See CMake documentation to see how to configure options

+ `CMAKE_INSTALL_PREFIX` - install the library to a local directory prefix
+ `BUILD_DOCS` - build the project documentation
+ `BUILD_TESTS` - build the test cases

## Building and Installing LibPressio

To build and install the library only.

```bash
BUILD_DIR=build
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake ..
make
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

To build the experimental python bindings:

```
BUILD_DIR=build
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake .. -DBUILD_PYTHON_WRAPPER=ON
make
make install
```

To build the test cases

```
BUILD_DIR=build
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake .. -DBUILD_TESTS=ON
make
ctest .
```

## Using LibPressio

Here is a minimal example with error checking of how to use LibPressio:


~~~c
#include <libpressio.h>
#include <libpressio_ext/compressors/sz.h>

#include "make_input_data.h"

int
main(int argc, char* argv[])
{
  struct pressio* library = pressio_instance();
  struct pressio_compressor* compressor = pressio_get_compressor(library, "sz");
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

  // free the input, decompressed, and compressed data
  pressio_data_free(decompressed_data);
  pressio_data_free(compressed_data);
  pressio_data_free(input_data);

  // free options and the library
  pressio_options_free(sz_options);
  pressio_release(&library);
  return 0;
}
~~~

More examples can be found in `test/`

## Option Names

LibPressio uses a key-value system to refer to configuration settings.

Each compressor may find specific configuration settings for its specific compressor with settings beginning with its compressor id as prefix (i.e. configurations for SZ begin with `sz:`).  Refer to the specific compressors documentation for further documentation for each settings.

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
