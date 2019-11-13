#include <stdio.h>
#include <iostream>
#include <cassert>
#include <libpressio.h>
#include <libpressio_ext/io/posix.h>
#include <libpressio_ext/cpp/printers.h>

#include "make_input_data.h"


int main(int argc, char *argv[])
{
  struct pressio* library = pressio_instance();
  struct pressio_compressor* compressor = pressio_get_compressor(library, "magick");
  struct pressio_options* magick_options = pressio_compressor_get_options(compressor);

  pressio_options_set_string(magick_options, "magick:compressed_magick", "JPEG");
  pressio_options_set_uinteger(magick_options, "magick:quality", 100);
  if(pressio_compressor_check_options(compressor, magick_options)) {
    printf("%s\n", pressio_compressor_error_msg(compressor));
    exit(pressio_compressor_error_code(compressor));
  }
  if(pressio_compressor_set_options(compressor, magick_options)) {
    printf("%s\n", pressio_compressor_error_msg(compressor));
    exit(pressio_compressor_error_code(compressor));
  }
  const char* metric_ids[] = {"error_stat"};
  pressio_metrics* metrics = pressio_new_metrics(library, metric_ids, 1);
  pressio_compressor_set_metrics(compressor, metrics);
  
  double rawinput_data[300][300];
  for (int j = 0; j < 300; ++j) {
    for (int i = 0; i < 300; ++i) {
      rawinput_data[j][i] = (i*3.1 + j*j)/(3.1*300.0+300.0*300);
    }
  }
  size_t dims[] = {300,300};
  struct pressio_data* input_data = pressio_data_new_nonowning(pressio_double_dtype, rawinput_data, 2, dims);

  //creates an output dataset pointer
  struct pressio_data* compressed_data = pressio_data_new_empty(pressio_byte_dtype, 0, NULL);
  struct pressio_data* recompressed_data = pressio_data_new_empty(pressio_byte_dtype, 0, NULL);

  //configure the decompressed output area
  struct pressio_data* decompressed_data = pressio_data_new_owning(pressio_double_dtype, 2, dims);

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

  pressio_options* metric_results = pressio_compressor_get_metrics_results(compressor);
  std::cout << *metric_results << std::endl;

  //recompress the data
  if(pressio_compressor_compress(compressor, decompressed_data, recompressed_data)) {
    printf("%s\n", pressio_compressor_error_msg(compressor));
    exit(pressio_compressor_error_code(compressor));
  }



  //free the input, decompressed, and compressed data
  pressio_data_free(recompressed_data);
  pressio_data_free(decompressed_data);
  pressio_data_free(compressed_data);
  pressio_data_free(input_data);

  //free options and the library
  pressio_options_free(metric_results);
  pressio_options_free(magick_options);
  pressio_metrics_free(metrics);
  pressio_compressor_release(compressor);
  pressio_release(library);
  return 0;
}
