#include "lossy_compressor_impl.h"

extern "C" {

struct lossy_options* lossy_compressor_get_options(struct lossy_compressor const* compressor) {
  return compressor->plugin->get_options();
}
int lossy_compressor_set_options(struct lossy_compressor* compressor, struct lossy_options const * options) {
  return compressor->plugin->set_options(options);
}
int lossy_compressor_compress(struct lossy_compressor* compressor, struct lossy_data * input, struct lossy_data ** output) {
  return compressor->plugin->compress(input, output);
}
int lossy_compressor_decompress(struct lossy_compressor* compressor, struct lossy_data * input, struct lossy_data ** output) {
  return compressor->plugin->decompress(input, output);
}
const char* lossy_compressor_version(struct lossy_compressor const* compressor) {
  return compressor->plugin->version();
}
int lossy_compressor_major_version(struct lossy_compressor const* compressor) {
  return compressor->plugin->major_version();
}
int lossy_compressor_minor_version(struct lossy_compressor const* compressor) {
  return compressor->plugin->minor_version();
}
int lossy_compressor_patch_version(struct lossy_compressor const* compressor) {
  return compressor->plugin->patch_version();
}
int lossy_compressor_error_code(struct lossy_compressor const* compressor) {
  return compressor->plugin->error_code();
}
const char* lossy_compressor_error_msg(struct lossy_compressor const* compressor) {
  return compressor->plugin->error_msg();
}
int lossy_compressor_check_options(struct lossy_compressor* compressor, struct lossy_options const * options) {
  return compressor->plugin->check_options(options);
}

}
