#include <liblossy.h>
#include <liblossy_ext/compressor_sz.h>

#include "make_input_data.h"


int main(int argc, char *argv[])
{
  struct lossy* library = lossy_instance();
  struct lossy_compressor* compressor = lossy_get_compressor(library, "zfp");
  struct lossy_options* sz_options = lossy_compressor_get_options(compressor);

  lossy_options_set_double(sz_options, "zfp:accuracy", 0.5);
  if(lossy_compressor_check_options(compressor, sz_options)) {
    printf("%s\n", lossy_compressor_error_msg(compressor));
    exit(lossy_compressor_error_code(compressor));
  }
  if(lossy_compressor_set_options(compressor, sz_options)) {
    printf("%s\n", lossy_compressor_error_msg(compressor));
    exit(lossy_compressor_error_code(compressor));
  }
  
  //load a 300x300x300 dataset into data created with malloc
  double* rawinput_data = make_input_data();
  size_t dims[] = {300,300,300};
  struct lossy_data* input_data = lossy_data_new_move(lossy_double_dtype, rawinput_data, 3, dims, lossy_data_libc_free_fn, NULL);

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

  //free the input, decompressed, and compressed data
  lossy_data_free(decompressed_data);
  lossy_data_free(compressed_data);
  lossy_data_free(input_data);

  //free options and the library
  lossy_options_free(sz_options);
  lossy_release(&library);
  return 0;
}
