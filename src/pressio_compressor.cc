#include "pressio_compressor_impl.h"

extern "C" {

struct pressio_options* pressio_compressor_get_options(struct pressio_compressor const* compressor) {
  return compressor->plugin->get_options();
}
int pressio_compressor_set_options(struct pressio_compressor* compressor, struct pressio_options const * options) {
  return compressor->plugin->set_options(options);
}
int pressio_compressor_compress(struct pressio_compressor* compressor, struct pressio_data * input, struct pressio_data ** output) {
  return compressor->plugin->compress(input, output);
}
int pressio_compressor_decompress(struct pressio_compressor* compressor, struct pressio_data * input, struct pressio_data ** output) {
  return compressor->plugin->decompress(input, output);
}
const char* pressio_compressor_version(struct pressio_compressor const* compressor) {
  return compressor->plugin->version();
}
int pressio_compressor_major_version(struct pressio_compressor const* compressor) {
  return compressor->plugin->major_version();
}
int pressio_compressor_minor_version(struct pressio_compressor const* compressor) {
  return compressor->plugin->minor_version();
}
int pressio_compressor_patch_version(struct pressio_compressor const* compressor) {
  return compressor->plugin->patch_version();
}
int pressio_compressor_error_code(struct pressio_compressor const* compressor) {
  return compressor->plugin->error_code();
}
const char* pressio_compressor_error_msg(struct pressio_compressor const* compressor) {
  return compressor->plugin->error_msg();
}
int pressio_compressor_check_options(struct pressio_compressor* compressor, struct pressio_options const * options) {
  return compressor->plugin->check_options(options);
}

}
