#include <assert.h>
#include <stdio.h>

#include <libpressio.h>
#include <sz.h>

#include "make_input_data.h"


void print_all_options(struct pressio_options* options) {
  struct pressio_options_iter* iter = pressio_options_get_iter(options);
  while(pressio_options_iter_has_value(iter)) 
  {
    char const* key = pressio_options_iter_get_key(iter);
    struct pressio_option* option = pressio_options_iter_get_value(iter);

    bool has_value = pressio_option_has_value(option);
    enum pressio_option_type otype = pressio_option_get_type(option);
    if(has_value) {
      switch(otype) {
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
        case pressio_option_data_type:
          printf("%s : pressio_data\n", key);
          break;
        case pressio_option_charptr_array_type:
          printf("%s <char**>: {", key);
          size_t size;
          const char** strings = pressio_option_get_strings(option, &size);
          for (size_t i = 0; i < size; ++i) {
            printf("%s, ", strings[i]);
            free((void*)strings[i]);
          }
          printf("}\n");
          free((void*)strings);
          break;
        default:
          assert(false && "a cleared option can never have a value");
      } 
    } else {
      switch(otype) {
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

void rescale(struct pressio_data* data) {
  assert(pressio_data_dtype(data) == pressio_double_dtype);
  double* begin_ptr = pressio_data_ptr(data, NULL);
  size_t size = pressio_data_num_elements(data);

  double min, max;
  min = max = begin_ptr[0];
  for (size_t i = 0; i < size; ++i) {
    if(begin_ptr[i] < min)  min = begin_ptr[i];
    if(begin_ptr[i] > max)  max = begin_ptr[i];
  }
  for (size_t i = 0; i < size; ++i) {
    begin_ptr[i] = (begin_ptr[i] - min)/ (max - min);
  }
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

void run_compressor(const char* compressor_name,
    struct pressio_compressor* compressor,
    struct pressio_options* options,
    struct pressio_data const* input_data
    ) {
  printf("%s\n configuration\n", compressor_name);
  //configure metrics
  const char* metrics[] = {"time", "size", "error_stat"};
  struct pressio* library = pressio_instance();
  struct pressio_metrics* metrics_plugin = pressio_new_metrics(library, metrics, 3);
  pressio_compressor_set_metrics(compressor, metrics_plugin);

  pressio_compressor_set_options(compressor, options);
  struct pressio_options* configuration = pressio_compressor_get_configuration(compressor);
  print_all_options(configuration);
  pressio_options_free(configuration);


  //creates an output dataset pointer
  struct pressio_data* compressed_data = pressio_data_new_empty(pressio_byte_dtype, 0, NULL);
  struct pressio_data* decompressed_data = pressio_data_new_clone(input_data);

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

  struct pressio_options* metrics_results = pressio_compressor_get_metrics_results(compressor);
  printf(" metrics results\n");
  print_all_options(metrics_results);

  pressio_data_free(decompressed_data);
  pressio_data_free(compressed_data);
  pressio_options_free(metrics_results);
  pressio_metrics_free(metrics_plugin);
  pressio_release(library);


  printf("done %s\n", compressor_name);
}

int main(int argc, char *argv[])
{
  //get an instance to the library
  struct pressio* library = pressio_instance();
  assert(library != NULL);

  //check if libpressio supports SZ
  printf("libpressio version: %s\n", pressio_version());
  assert(strstr(pressio_features(), "sz") != NULL);
  assert(strstr(pressio_features(), "zfp") != NULL);
  assert(strstr(pressio_features(), "mgard") != NULL);
  assert(strstr(pressio_features(), "magick") != NULL);
  assert(strstr(pressio_features(), "blosc") != NULL);
  assert(strstr(pressio_features(), "fpzip") != NULL);

  //get an instance of a compressor
  struct pressio_compressor* zfp_compressor = pressio_get_compressor(library, "zfp");
  struct pressio_compressor* sz_compressor = pressio_get_compressor(library, "sz");
  struct pressio_compressor* mgard_compressor = pressio_get_compressor(library, "mgard");
  struct pressio_compressor* magick_compressor = pressio_get_compressor(library, "magick");
  struct pressio_compressor* blosc_compressor = pressio_get_compressor(library, "blosc");
  struct pressio_compressor* fpzip_compressor = pressio_get_compressor(library, "fpzip");
  struct pressio_compressor* sample_compressor = pressio_get_compressor(library, "sample");

  assert(sz_compressor != NULL);
  assert(zfp_compressor != NULL);
  assert(mgard_compressor != NULL);
  assert(magick_compressor != NULL);
  assert(fpzip_compressor != NULL);
  assert(sample_compressor != NULL);

  printf("sz version: %s\n", pressio_compressor_version(sz_compressor));
  printf("zfp version: %s\n", pressio_compressor_version(zfp_compressor));
  printf("mgard version: %s\n", pressio_compressor_version(mgard_compressor));
  printf("blosc version: %s\n", pressio_compressor_version(blosc_compressor));
  printf("fpzip version: %s\n", pressio_compressor_version(fpzip_compressor));
  printf("sample version: %s\n", pressio_compressor_version(sample_compressor));

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

  struct pressio_options* mgard_options = pressio_compressor_get_options(mgard_compressor);
  pressio_options_set_double(mgard_options, "mgard:s", 0.5);
  pressio_options_set_double(mgard_options, "mgard:tolerance", .5);
  if(pressio_compressor_check_options(mgard_compressor, mgard_options)) {
    printf("%s\n", pressio_compressor_error_msg(mgard_compressor));
    exit(pressio_compressor_error_code(mgard_compressor));
  }

  struct pressio_options* magick_options = pressio_compressor_get_options(magick_compressor);
  pressio_options_set_string(magick_options, "magick:compressed_magick", "GIF");
  if(pressio_compressor_check_options(magick_compressor, magick_options)) {
    printf("%s\n", pressio_compressor_error_msg(magick_compressor));
    exit(pressio_compressor_error_code(magick_compressor));
  }

  struct pressio_options* blosc_options = pressio_compressor_get_options(blosc_compressor);
  pressio_options_set_integer(blosc_options, "blosc:clevel", 9);
  if(pressio_compressor_check_options(blosc_compressor, blosc_options)) {
    printf("%s\n", pressio_compressor_error_msg(blosc_compressor));
    exit(pressio_compressor_error_code(blosc_compressor));
  }

  struct pressio_options* fpzip_options = pressio_compressor_get_options(fpzip_compressor);
  pressio_options_set_uinteger(blosc_options, "fpzip:prec", 4);
  if(pressio_compressor_check_options(fpzip_compressor, fpzip_options)) {
    printf("%s\n", pressio_compressor_error_msg(fpzip_compressor));
    exit(pressio_compressor_error_code(fpzip_compressor));
  }

  struct pressio_options* sampling_options = pressio_compressor_get_options(sample_compressor);
  pressio_options_set_string(sampling_options, "sample:mode", "wor");
  pressio_options_set_double(sampling_options, "sample:rate", .25);
  pressio_options_set_integer(sampling_options, "sample:seed", 0);
  if(pressio_compressor_check_options(sample_compressor, sampling_options)) {
    printf("%s\n", pressio_compressor_error_msg(sample_compressor));
    exit(pressio_compressor_error_code(sample_compressor));
  }




  struct pressio_options* options_part1 = pressio_options_merge(sz_options, zfp_options);
  struct pressio_options* options_part2 = pressio_options_merge(options_part1, mgard_options);
  struct pressio_options* options_part3 = pressio_options_merge(options_part2, blosc_options);
  struct pressio_options* options_part4 = pressio_options_merge(options_part3, fpzip_options);
  struct pressio_options* options_part5 = pressio_options_merge(options_part4, sampling_options);
  struct pressio_options* options = pressio_options_merge(options_part5, magick_options);
  pressio_options_free(options_part1);
  pressio_options_free(options_part2);
  pressio_options_free(options_part3);
  pressio_options_free(options_part4);
  pressio_options_free(options_part5);
  pressio_options_free(sz_options);
  pressio_options_free(zfp_options);
  pressio_options_free(mgard_options);
  pressio_options_free(magick_options);
  pressio_options_free(blosc_options);
  pressio_options_free(fpzip_options);
  pressio_options_free(sampling_options);
  assert(options != NULL);

  //print all the configuration options
  print_all_options(options);
  test_options(options);

  //load a 300x300x300 dataset
  double* rawinput_data = make_input_data();
  size_t dims[] = {300,300,300};
  struct pressio_data* input_data = pressio_data_new_move(pressio_double_dtype, rawinput_data, 3, dims, pressio_data_libc_free_fn, NULL);


  //select the first image for magick2d
  size_t first_layer_dims[] = {1,300,300};
  struct pressio_data* first_layer = pressio_data_select(input_data, /*start*/NULL, /*stride*/NULL, first_layer_dims, /*blocks*/NULL);
  pressio_data_reshape(first_layer, 2, &first_layer_dims[1]);
  rescale(first_layer);


  run_compressor("sampling", sample_compressor, options, input_data);
  run_compressor("sz", sz_compressor, options, input_data);
  run_compressor("zfp", zfp_compressor, options, input_data);
  run_compressor("mgard", mgard_compressor, options, input_data);
  run_compressor("blosc", blosc_compressor, options, input_data);
  run_compressor("fpzip", fpzip_compressor, options, input_data);
  run_compressor("magick 2d", magick_compressor, options, first_layer);
  rescale(input_data);
  run_compressor("magick 3d", magick_compressor, options, input_data);

  pressio_options_free(options);
  pressio_data_free(input_data);
  pressio_data_free(first_layer);
  pressio_compressor_release(sz_compressor);
  pressio_compressor_release(zfp_compressor);
  pressio_compressor_release(mgard_compressor);
  pressio_compressor_release(magick_compressor);
  pressio_compressor_release(blosc_compressor);
  pressio_compressor_release(fpzip_compressor);
  pressio_compressor_release(sample_compressor);
  pressio_release(library);

  return 0;
}
