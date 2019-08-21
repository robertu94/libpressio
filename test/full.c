#include <assert.h>
#include <stdio.h>

#include <pressio.h>
#include <pressio_compressor.h>
#include <pressio_options.h>
#include <pressio_options_iter.h>
#include <pressio_option.h>
#include <pressio_data.h>
#include <libpressio_ext/compressors/sz.h>

#include "make_input_data.h"

void print_all_options(struct pressio_options* options) {
  struct pressio_options_iter* iter = pressio_options_get_iter(options);
  while(pressio_options_iter_has_value(iter)) 
  {
    char const* key = pressio_options_iter_get_key(iter);
    struct pressio_option* option = pressio_options_iter_get_value(iter);

    if(pressio_option_has_value(option)) {
      switch(pressio_option_get_type(option)) {
        case pressio_option_charptr_type:
          printf("%s : \"%s\"\n", key, pressio_option_get_string(option));
          break;
        case pressio_option_userptr_type:
          printf("%s : %p\n", key, pressio_option_get_userptr(option));
          break;
        case pressio_option_int32_type:
          printf("%s : %d\n", key, pressio_option_get_integer(option));
          break;
        case pressio_option_uint32_type:
          printf("%s : %u\n", key, pressio_option_get_uinteger(option));
          break;
        case pressio_option_double_type:
          printf("%s : %lf\n", key, pressio_option_get_double(option));
          break;
        case pressio_option_float_type:
          printf("%s : %f\n", key, pressio_option_get_float(option));
          break;
        default:
          assert(false && "a cleared option can never have a value");
      } 
    } else {
      switch(pressio_option_get_type(option)) {
        case pressio_option_charptr_type:
          printf("%s <string>: null\n", key);
          break;
        case pressio_option_userptr_type:
          printf("%s <void*>: null\n", key);
          break;
        case pressio_option_int32_type:
          printf("%s <int32_t>: null\n", key);
          break;
        case pressio_option_uint32_type:
          printf("%s <uint32_t>: null\n", key);
          break;
        case pressio_option_double_type:
          printf("%s <double>: null\n", key);
          break;
        case pressio_option_float_type:
          printf("%s <float>: null\n", key);
          break;
        case pressio_option_unset_type:
          printf("%s <unset>: null\n", key);
          break;
      }
    }

    pressio_option_free(option);
    pressio_options_iter_next(iter);
  }
  pressio_options_iter_free(iter);
}

void test_options(struct pressio_options const* options) {
  struct pressio_options* sz_options = pressio_options_copy(options);
  //set a configuration option for a compressor
  pressio_options_set_integer(sz_options, "sz:mode", ABS);

  //get the same configuration option back out
  struct pressio_option* mode_option = pressio_options_get(sz_options, "sz:mode");
  enum pressio_option_type type = pressio_option_get_type(mode_option);
  int mode = pressio_option_get_integer(mode_option);
  assert(mode == ABS);
  assert(type == pressio_option_int32_type);

  //a simpler way to get options back out
  int mode2;
  if(pressio_options_get_integer(sz_options, "sz:mode", &mode2))
  {
    assert(false && "the mode should be defined");
  }
  pressio_option_free(mode_option);
  pressio_options_free(sz_options);
}

void run_compressor(const char* compressor_name, struct pressio_compressor* compressor, struct pressio_options* options) {
  printf("%s\n", compressor_name);
  pressio_compressor_set_options(compressor, options);


  //load a 300x300x300 dataset
  double* rawinput_data = make_input_data();
  size_t dims[] = {300,300,300};
  struct pressio_data* input_data = pressio_data_new_move(pressio_double_dtype, rawinput_data, 3, dims, pressio_data_libc_free_fn, NULL);

  //creates an output dataset pointer
  struct pressio_data* compressed_data = pressio_data_new_empty(pressio_byte_dtype, 0, NULL);
  struct pressio_data* decompressed_data = pressio_data_new_empty(pressio_double_dtype, 3, dims);

  //compress the data
  if(pressio_compressor_compress(compressor, input_data, compressed_data)) {
    printf("%s\n", pressio_compressor_error_msg(compressor));
    exit(pressio_compressor_error_code(compressor));
  }
  
  //decompress the data
  if(pressio_compressor_decompress(compressor, compressed_data, decompressed_data)) {
    printf("%s\n", pressio_compressor_error_msg(compressor));
    exit(pressio_compressor_error_code(compressor));
  }

  pressio_data_free(decompressed_data);
  pressio_data_free(compressed_data);
  pressio_data_free(input_data);


}

int main(int argc, char *argv[])
{
  //get an instance to the library
  struct pressio* library = pressio_instance();
  assert(library != NULL);

  //check if libpressio supports SZ
  printf("libpressio version: %s\n", pressio_version());
  assert(strstr(pressio_features(), "sz") != NULL);

  //get an instance of a compressor
  struct pressio_compressor* zfp_compressor = pressio_get_compressor(library, "zfp");
  struct pressio_compressor* sz_compressor = pressio_get_compressor(library, "sz");
  printf("sz version: %s\n", pressio_compressor_version(sz_compressor));
  printf("zfp version: %s\n", pressio_compressor_version(zfp_compressor));
  assert(sz_compressor != NULL);

  //get the list of configuration options
  printf("getting options\n");
  struct pressio_options* sz_options = pressio_compressor_get_options(sz_compressor);
  pressio_options_set_double(sz_options, "sz:abs_err_bound", .5);
  if(pressio_compressor_check_options(sz_compressor, sz_options)) {
    printf("%s\n", pressio_compressor_error_msg(sz_compressor));
    exit(pressio_compressor_error_code(sz_compressor));
  }

  struct pressio_options* zfp_options = pressio_compressor_get_options(zfp_compressor);
  pressio_options_set_double(zfp_options, "zfp:accuracy", .5);
  if(pressio_compressor_check_options(zfp_compressor, zfp_options)) {
    printf("%s\n", pressio_compressor_error_msg(zfp_compressor));
    exit(pressio_compressor_error_code(zfp_compressor));
  }

  struct pressio_options* options = pressio_options_merge(sz_options, zfp_options);
  pressio_options_free(sz_options);
  pressio_options_free(zfp_options);
  assert(options != NULL);

  //print all the configuration options
  print_all_options(options);
  test_options(options);

  run_compressor("sz", sz_compressor, options);
  run_compressor("zfp", zfp_compressor, options);

  pressio_options_free(options);
  pressio_release(&library);

  return 0;
}
