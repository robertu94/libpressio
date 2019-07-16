# Liblossy

Liblossy is a library that abstracts differences between different lossless and lossy compressors.

## Configuring Liblossy

Liblossy uses cmake to configure build options.  See CMake documentation to see how to configure options

+ `CMAKE_INSTALL_PREFIX` - install the library to a local directory prefix
+ `BUILD_DOCS` - build the project documentation
+ `BUILD_TESTS` - build the test cases

## Building and Installing Liblossy

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
cmake . -DBUILD_DOCS=ON
make docs
# the html docs can be found in $BUILD_DIR/html/index.html
# the man pages can be found in $BUILD_DIR/man/
```

## Using Liblossy

Here is a minimal example of how to use liblossy:


~~~{.c}
#include <liblossy.h>
#include <liblossy_ext/compressor_sz.h>

#include "make_input_data.h"

int main(int argc, char *argv[])
{
  struct lossy* library = lossy_instance();
  struct lossy_compressor* compressor = lossy_get_compressor(library, "sz");
  struct lossy_options* sz_options = lossy_compressor_get_options(compressor);

  lossy_options_set_integer(sz_options, "sz:mode", ABS);
  lossy_options_set_double(sz_options, "sz:abs_error_bound", 0.5);
  lossy_compressor_set_options(compressor, sz_options);
  
  //load a 300x300x300 dataset
  double* rawinput_data = make_input_data();
  size_t dims[] = {300,300,300};
  struct lossy_data* input_data = lossy_data_new(lossy_double_dtype, rawinput_data, 3, dims);

  //creates an output dataset pointer
  struct lossy_data* compressed_data = lossy_data_new_empty(lossy_byte_dtype, 0, NULL);

  //configure the decompressed output area
  struct lossy_data* decompressed_data = lossy_data_new_empty(lossy_double_dtype, 3, dims);

  //compress the data
  if(lossy_compressor_compress(compressor, input_data, &compressed_data)) {
    printf("%s\n", lossy_compressor_error_msg(compressor));
    exit(lossy_compressor_error_code(compressor));
  }
  
  //decompress the data
  if(lossy_compressor_decompress(compressor, compressed_data, &decompressed_data)) {
    printf("%s\n", lossy_compressor_error_msg(compressor));
    exit(lossy_compressor_error_code(compressor));
  }

  //free the decompressed_data
  free(lossy_data_ptr(decompressed_data, NULL));
  lossy_data_free(decompressed_data);

  //free the compressed_data
  free(lossy_data_ptr(compressed_data, NULL));
  lossy_data_free(compressed_data);

  //free the raw input data
  free(rawinput_data);
  lossy_data_free(input_data);

  //free options and the library
  lossy_options_free(sz_options);
  lossy_release(&library);
  return 0;
}
~~~

More examples can be found in `test/`
