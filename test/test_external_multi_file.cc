#include <functional>
#include <iostream>
#include <libpressio.h>
#include <libpressio_ext/cpp/printers.h>
#include <numeric>
#include <cassert>

pressio_data* make_input_data_const(int32_t value) {
  size_t dims[] = {30,30,30};
  std::vector<int32_t> v(std::accumulate(std::begin(dims), std::end(dims), 1, compat::multiplies<>{}), value);

  return pressio_data_new_copy(pressio_int32_dtype, v.data(), 3, dims);
}


int main(int argc, char *argv[])
{
  struct pressio* library = pressio_instance();
  struct pressio_compressor* compressor = pressio_get_compressor(library, "noop");

  const char* metrics[] = {"external"};
  struct pressio_metrics* metrics_plugin = pressio_new_metrics(library, metrics, 1);
  pressio_compressor_set_metrics(compressor, metrics_plugin);

  struct pressio_options* external_options = pressio_compressor_metrics_get_options(compressor);
  assert(argc==2);
  const char* formats[] = {"posix", "posix"};
  const char* fieldnames[] = {"first", "second"};
  pressio_options_set_string(external_options, "external:command", argv[1]);
  pressio_options_set_strings(external_options, "external:io_format", 2, formats);
  pressio_options_set_strings(external_options, "external:fieldnames", 2, fieldnames);
  pressio_compressor_metrics_set_options(compressor, external_options);

  
  //load a 300x300x300 dataset into data created with malloc
  struct pressio_data* first_input_data = make_input_data_const(1);

  //creates an output dataset pointer
  struct pressio_data* first_compressed_data = pressio_data_new_empty(pressio_byte_dtype, 0, NULL);

  //configure the decompressed output area
  struct pressio_data* first_decompressed_data = make_input_data_const(1);

  
  //load a 300x300x300 dataset into data created with malloc
  struct pressio_data* second_input_data = make_input_data_const(2);

  //creates an output dataset pointer
  struct pressio_data* second_compressed_data = pressio_data_new_empty(pressio_byte_dtype, 0, NULL);

  //configure the decompressed output area
  struct pressio_data* second_decompressed_data = make_input_data_const(2);

  pressio_data* input_datas[] = {first_input_data, second_input_data};
  pressio_data* compressed_datas[] = {first_compressed_data, second_compressed_data};
  pressio_data* decompressed_datas[] = {first_compressed_data, second_compressed_data};

  //compress the data
  if(pressio_compressor_compress_many(compressor, input_datas, 2, compressed_datas, 2)) {
    printf("%s\n", pressio_compressor_error_msg(compressor));
    exit(pressio_compressor_error_code(compressor));
  }
  
  //decompress the data
  if(pressio_compressor_decompress_many(compressor, compressed_datas, 2, decompressed_datas, 2)) {
    printf("%s\n", pressio_compressor_error_msg(compressor));
    exit(pressio_compressor_error_code(compressor));
  }
  
  struct pressio_options* metrics_result = pressio_compressor_get_metrics_results(compressor);
  std::cout << *metrics_result << std::endl;

  int error_code = -1, return_code = -1;
  const char* errors = nullptr;
  double observed_defaulted=0, observed_defaulted2=0, observed_first_dims=0, observed_second_dims=0;
  pressio_options_get_integer(metrics_result, "external:error_code", &error_code);
  pressio_options_get_integer(metrics_result, "external:return_code", &return_code);
  pressio_options_get_string(metrics_result, "external:stderr", &errors);
  pressio_options_get_double(metrics_result, "external:results:defaulted", &observed_defaulted);
  pressio_options_get_double(metrics_result, "external:results:defaulted2", &observed_defaulted2);
  pressio_options_get_double(metrics_result, "external:results:first_dims", &observed_first_dims);
  pressio_options_get_double(metrics_result, "external:results:first_dims", &observed_second_dims);

  if(error_code) {
    printf("FAILURE: unexpected non-zero error code %d\n", error_code);
    exit(error_code);
  }

  if(return_code) {
    printf("FAILURE: unexpected non-zero return code %d\n", return_code);
    exit(return_code);
  }
  if(!errors) {
    printf("FAILURE: no warning text gathered\n");
    exit(1);
  }
  if(!strstr(errors, "testing warning")) {
    printf("FAILURE: failed to find expected warning\n");
    exit(1);
  }
  free(const_cast<char*>(errors));

  if(observed_defaulted != 2.0) {
    printf("FAILURE: wrong value for defaulted %lg\n", observed_defaulted);
    exit(1);
  }
  if(observed_defaulted2 != 17.1) {
    printf("FAILURE: wrong value for defaulted2 %lg\n", observed_defaulted2);
    exit(1);
  }
  if(observed_first_dims != 3.0) {
    printf("FAILURE: wrong value for dims %lg\n", observed_first_dims);
    exit(1);
  }
  if(observed_second_dims != 3.0) {
    printf("FAILURE: wrong value for dims %lg\n", observed_second_dims);
    exit(1);
  }



  //free the input, decompressed, and compressed data
  pressio_data_free(first_decompressed_data);
  pressio_data_free(first_compressed_data);
  pressio_data_free(first_input_data);
  pressio_data_free(second_decompressed_data);
  pressio_data_free(second_compressed_data);
  pressio_data_free(second_input_data);

  //free options and the library
  pressio_metrics_free(metrics_plugin);
  pressio_options_free(metrics_result);
  pressio_options_free(external_options);
  pressio_compressor_release(compressor);
  pressio_release(library);
  return 0;
}
