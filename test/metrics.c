#include <libpressio.h>
#include <sz.h>

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
        case pressio_option_int16_type:
          printf("%s : %hd\n", key, pressio_option_get_integer16(option));
          break;
        case pressio_option_uint16_type:
          printf("%s : %hu\n", key, pressio_option_get_uinteger16(option));
          break;
        case pressio_option_int32_type:
          printf("%s : %d\n", key, pressio_option_get_integer(option));
          break;
        case pressio_option_uint32_type:
          printf("%s : %u\n", key, pressio_option_get_uinteger(option));
          break;
        case pressio_option_int64_type:
          printf("%s : %ld\n", key, pressio_option_get_integer64(option));
          break;
        case pressio_option_uint64_type:
          printf("%s : %lu\n", key, pressio_option_get_uinteger64(option));
          break;
        case pressio_option_double_type:
          printf("%s : %lf\n", key, pressio_option_get_double(option));
          break;
        case pressio_option_float_type:
          printf("%s : %f\n", key, pressio_option_get_float(option));
          break;
        case pressio_option_bool_type:
          printf("%s : %s\n", key, pressio_option_get_bool(option) ? "true":"false");
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
        case pressio_option_int8_type:
          printf("%s <int8_t>: null\n", key);
          break;
        case pressio_option_uint8_type:
          printf("%s <uint8_t>: null\n", key);
          break;
        case pressio_option_int16_type:
          printf("%s <int16_t>: null\n", key);
          break;
        case pressio_option_uint16_type:
          printf("%s <uint16_t>: null\n", key);
          break;
        case pressio_option_int32_type:
            printf("%s <int32_t>: null\n", key);
            break;
        case pressio_option_uint32_type:
          printf("%s <uint32_t>: null\n", key);
          break;
        case pressio_option_int64_type:
          printf("%s <int64_t>: null\n", key);
          break;
        case pressio_option_uint64_type:
          printf("%s <uint64_t>: null\n", key);
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
        case pressio_option_charptr_array_type:
          printf("%s <char**>: null\n", key);
          break;
        case pressio_option_data_type:
          printf("%s <data>: null\n", key);
          break;
        case pressio_option_bool_type:
          printf("%s <bool>: null\n", key);
          break;
      }
    }

    pressio_option_free(option);
    pressio_options_iter_next(iter);
  }
  pressio_options_iter_free(iter);
}

int main()
{
  struct pressio* library = pressio_instance();
  struct pressio_compressor* compressor = pressio_get_compressor(library, "sz");

  const char* metrics[] = {"time", "size", "error_stat"};
  struct pressio_metrics* metrics_plugin = pressio_new_metrics(library, metrics, 3);
  pressio_compressor_set_metrics(compressor, metrics_plugin);

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
  double*raw_input_data = make_input_data();
  size_t dims[] = {300,300,300};
  struct pressio_data* input_data = pressio_data_new_move(pressio_double_dtype, raw_input_data, 3, dims, pressio_data_libc_free_fn, NULL);

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
  print_all_options(metrics_result);


  //free the input, decompressed, and compressed data
  pressio_data_free(decompressed_data);
  pressio_data_free(compressed_data);
  pressio_data_free(input_data);

  //free options and the library
  pressio_metrics_free(metrics_plugin);
  pressio_options_free(metrics_result);
  pressio_options_free(sz_options);
  pressio_compressor_release(compressor);
  pressio_release(library);
  return 0;
}
