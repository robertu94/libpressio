#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"

extern "C" {

struct pressio_options* pressio_compressor_get_configuration(struct pressio_compressor const* compressor) {
  return new pressio_options((*compressor)->get_configuration());
}
struct pressio_options* pressio_compressor_get_options(struct pressio_compressor const* compressor) {
  return new pressio_options((*compressor)->get_options());
}
int pressio_compressor_set_options(struct pressio_compressor* compressor, struct pressio_options const * options) {
  return (*compressor)->set_options(*options);
}
int pressio_compressor_compress(struct pressio_compressor* compressor, const pressio_data *input, struct pressio_data * output) {
  return (*compressor)->compress(input, output);
}
int pressio_compressor_decompress(struct pressio_compressor* compressor, const pressio_data *input, struct pressio_data * output) {
  return (*compressor)->decompress(input, output);
}
const char* pressio_compressor_version(struct pressio_compressor const* compressor) {
  return (*compressor)->version();
}
int pressio_compressor_major_version(struct pressio_compressor const* compressor) {
  return (*compressor)->major_version();
}
int pressio_compressor_minor_version(struct pressio_compressor const* compressor) {
  return (*compressor)->minor_version();
}
int pressio_compressor_patch_version(struct pressio_compressor const* compressor) {
  return (*compressor)->patch_version();
}
int pressio_compressor_error_code(struct pressio_compressor const* compressor) {
  return (*compressor)->error_code();
}
const char* pressio_compressor_error_msg(struct pressio_compressor const* compressor) {
  return (*compressor)->error_msg();
}
int pressio_compressor_check_options(struct pressio_compressor* compressor, struct pressio_options const * options) {
  return (*compressor)->check_options(*options);
}

struct pressio_options* pressio_compressor_get_metrics_results(struct pressio_compressor const* compressor) {
  return new pressio_options((*compressor)->get_metrics_results());
}

struct pressio_metrics* pressio_compressor_get_metrics(struct pressio_compressor const* compressor) {
  return new pressio_metrics((*compressor)->get_metrics());
}

void pressio_compressor_set_metrics(struct pressio_compressor* compressor, struct pressio_metrics* plugin) {
  return (*compressor)->set_metrics(*plugin);
}

void pressio_compressor_release(struct pressio_compressor* compressor) {
  delete compressor;
}

struct pressio_options* pressio_compressor_metrics_get_options(struct pressio_compressor const* compressor) {
  return new pressio_options((*compressor)->get_metrics_options());
}

int pressio_compressor_metrics_set_options(struct pressio_compressor const* compressor, struct pressio_options const* options) {
  return (*compressor)->set_metrics_options(*options);
}

struct pressio_compressor* pressio_compressor_clone(struct pressio_compressor* compressor) {
  return new pressio_compressor((*compressor)->clone());
}

void pressio_compressor_set_name(struct pressio_compressor* compressor, const char* new_name) {
  (*compressor)->set_name(new_name);
}


const char* pressio_compressor_get_name(struct pressio_compressor const* compressor) {
  return (*compressor)->get_name().c_str();
}

int pressio_compressor_compress_many(struct pressio_compressor* compressor,
    struct pressio_data const* in[], size_t num_inputs,
    struct pressio_data * out[], size_t num_outputs
    ) {
  return (*compressor)->compress_many(in, in+num_inputs, out, out+num_outputs);
}

int pressio_compressor_decompress_many(struct pressio_compressor* compressor,
    struct pressio_data const* in[], size_t num_inputs,
    struct pressio_data * out[], size_t num_outputs
    ) {
  return (*compressor)->decompress_many(in, in+num_inputs, out, out+num_outputs);
}


}
