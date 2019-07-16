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

int main(int argc, char *argv[])
{
  //get an instance to the library
  struct lossy* library = lossy_instance();
  assert(library != NULL);

  //check if liblossy supports SZ
  printf("liblossy version: %s\n", lossy_version());
  assert(strstr(lossy_features(), "sz") != NULL);

  //get an instance of a compressor
  struct lossy_compressor* compressor = lossy_get_compressor(library, "sz");
  printf("sz version: %s\n", lossy_compressor_version(compressor));
  assert(compressor != NULL);

  //get the list of configuration options
  printf("getting options\n");
  struct lossy_options* sz_options = lossy_compressor_get_options(compressor);
  assert(sz_options != NULL);

  //print all the configuration options
  print_all_options(sz_options);

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

  printf("setting options\n");
  lossy_compressor_set_options(compressor, sz_options);


  //load a 300x300x300 dataset
  double* rawinput_data = make_input_data();
  size_t dims[] = {300,300,300};
  struct lossy_data* input_data = lossy_data_new(lossy_double_dtype, rawinput_data, 3, dims);

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

  free(lossy_data_ptr(decompressed_data, NULL));
  lossy_data_free(decompressed_data);

  free(lossy_data_ptr(compressed_data, NULL));
  lossy_data_free(compressed_data);

  free(rawinput_data);
  lossy_data_free(input_data);

  lossy_options_free(sz_options);
  lossy_option_free(mode_option);
  lossy_release(&library);

  return 0;
}
