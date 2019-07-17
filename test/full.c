#include <assert.h>
#include <stdio.h>

#include <lossy.h>
#include <lossy_compressor.h>
#include <lossy_options.h>
#include <lossy_options_iter.h>
#include <lossy_option.h>
#include <lossy_data.h>
#include <liblossy_ext/compressor_sz.h>

#include "make_input_data.h"

void print_all_options(struct lossy_options* options) {
  struct lossy_options_iter* iter = lossy_options_get_iter(options);
  while(lossy_options_iter_has_value(iter)) 
  {
    char const* key = lossy_options_iter_get_key(iter);
    struct lossy_option* option = lossy_options_iter_get_value(iter);

    switch(lossy_option_get_type(option)) {
      case lossy_option_charptr_type:
        printf("%s : \"%s\"\n", key, lossy_option_get_string(option));
        break;
      case lossy_option_userptr_type:
        printf("%s : %p\n", key, lossy_option_get_userptr(option));
        break;
      case lossy_option_int32_type:
        printf("%s : %d\n", key, lossy_option_get_integer(option));
        break;
      case lossy_option_uint32_type:
        printf("%s : %u\n", key, lossy_option_get_uinteger(option));
        break;
      case lossy_option_double_type:
        printf("%s : %lf\n", key, lossy_option_get_double(option));
        break;
      case lossy_option_float_type:
        printf("%s : %f\n", key, lossy_option_get_float(option));
        break;
      case lossy_option_unset:
        printf("%s : null\n", key);
        break;
    }

    lossy_option_free(option);
    lossy_options_iter_next(iter);
  }
  lossy_options_iter_free(iter);
}

void test_options(struct lossy_options const* options) {
  struct lossy_options* sz_options = lossy_options_copy(options);
  //set a configuration option for a compressor
  lossy_options_set_integer(sz_options, "sz:mode", ABS);

  //get the same configuration option back out
  struct lossy_option* mode_option = lossy_options_get(sz_options, "sz:mode");
  enum lossy_option_type type = lossy_option_get_type(mode_option);
  int mode = lossy_option_get_integer(mode_option);
  assert(mode == ABS);
  assert(type == lossy_option_int32_type);

  //a simpler way to get options back out
  int mode2;
  if(lossy_options_get_integer(sz_options, "sz:mode", &mode2))
  {
    assert(false && "the mode should be defined");
  }
  lossy_option_free(mode_option);
  lossy_options_free(sz_options);
}

void run_compressor(const char* compressor_name, struct lossy_compressor* compressor, struct lossy_options* options) {
  printf("%s\n", compressor_name);
  lossy_compressor_set_options(compressor, options);


  //load a 300x300x300 dataset
  double* rawinput_data = make_input_data();
  size_t dims[] = {300,300,300};
  struct lossy_data* input_data = lossy_data_new_move(lossy_double_dtype, rawinput_data, 3, dims, lossy_data_libc_free_fn, NULL);

  //creates an output dataset pointer
  struct lossy_data* compressed_data = lossy_data_new_empty(lossy_byte_dtype, 0, NULL);
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

  lossy_data_free(decompressed_data);
  lossy_data_free(compressed_data);
  lossy_data_free(input_data);


}

int main(int argc, char *argv[])
{
  //get an instance to the library
  struct lossy* library = lossy_instance();
  assert(library != NULL);

  //check if liblossy supports SZ
  printf("liblossy version: %s\n", lossy_version());
  assert(strstr(lossy_features(), "sz") != NULL);

  //get an instance of a compressor
  struct lossy_compressor* zfp_compressor = lossy_get_compressor(library, "zfp");
  struct lossy_compressor* sz_compressor = lossy_get_compressor(library, "sz");
  printf("sz version: %s\n", lossy_compressor_version(sz_compressor));
  printf("zfp version: %s\n", lossy_compressor_version(zfp_compressor));
  assert(sz_compressor != NULL);

  //get the list of configuration options
  printf("getting options\n");
  struct lossy_options* sz_options = lossy_compressor_get_options(sz_compressor);
  lossy_options_set_double(sz_options, "sz:abs_err_bound", .5);
  if(lossy_compressor_check_options(sz_compressor, sz_options)) {
    printf("%s\n", lossy_compressor_error_msg(sz_compressor));
    exit(lossy_compressor_error_code(sz_compressor));
  }

  struct lossy_options* zfp_options = lossy_compressor_get_options(zfp_compressor);
  lossy_options_set_double(zfp_options, "zfp:accuracy", .5);
  if(lossy_compressor_check_options(zfp_compressor, zfp_options)) {
    printf("%s\n", lossy_compressor_error_msg(zfp_compressor));
    exit(lossy_compressor_error_code(zfp_compressor));
  }

  struct lossy_options* options = lossy_options_merge(sz_options, zfp_options);
  lossy_options_free(sz_options);
  lossy_options_free(zfp_options);
  assert(options != NULL);

  //print all the configuration options
  print_all_options(options);
  test_options(options);

  run_compressor("sz", sz_compressor, options);
  run_compressor("zfp", zfp_compressor, options);

  lossy_options_free(options);
  lossy_release(&library);

  return 0;
}
