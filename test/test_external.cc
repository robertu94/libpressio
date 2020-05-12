#include <iostream>
#include <libpressio.h>
#include <libpressio_ext/cpp/printers.h>
#include <sz.h>

#include "make_input_data.h"


int main(int argc, char *argv[])
{
  struct pressio* library = pressio_instance();
  struct pressio_compressor* compressor = pressio_get_compressor(library, "sz");

  const char* metrics[] = {"external"};
  struct pressio_metrics* metrics_plugin = pressio_new_metrics(library, metrics, 1);
  pressio_compressor_set_metrics(compressor, metrics_plugin);

  struct pressio_options* external_options = pressio_compressor_metrics_get_options(compressor);
  assert(argc==2);
  pressio_options_set_string(external_options, "external:command", argv[1]);
  pressio_compressor_metrics_set_options(compressor, external_options);

  struct pressio_options* sz_options = pressio_compressor_get_options(compressor);

  pressio_options_set_integer(sz_options, "sz:error_bound_mode", ABS);
  pressio_options_set_double(sz_options, "sz:abs_err_bound", 0.05);
  if(pressio_compressor_check_options(compressor, sz_options)) {
    printf("%s\n", pressio_compressor_error_msg(compressor));
    exit(pressio_compressor_error_code(compressor));
  }
  if(pressio_compressor_set_options(compressor, sz_options)) {
    printf("%s\n", pressio_compressor_error_msg(compressor));
    exit(pressio_compressor_error_code(compressor));
  }

  
  //load a 300x300x300 dataset into data created with malloc
  double* rawinput_data = make_input_data();
  size_t dims[] = {30,30,30};
  struct pressio_data* input_data = pressio_data_new_move(pressio_double_dtype, rawinput_data, 3, dims, pressio_data_libc_free_fn, NULL);

  //creates an output dataset pointer
  struct pressio_data* compressed_data = pressio_data_new_empty(pressio_byte_dtype, 0, NULL);

  //configure the decompressed output area
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
  
  struct pressio_options* metrics_result = pressio_compressor_get_metrics_results(compressor);
  std::cout << *metrics_result << std::endl;


  //free the input, decompressed, and compressed data
  pressio_data_free(decompressed_data);
  pressio_data_free(compressed_data);
  pressio_data_free(input_data);

  //free options and the library
  pressio_metrics_free(metrics_plugin);
  pressio_options_free(metrics_result);
  pressio_options_free(sz_options);
  pressio_options_free(external_options);
  pressio_compressor_release(compressor);
  pressio_release(library);
  return 0;
}

