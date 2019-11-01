#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"

extern "C" {

struct pressio_options* pressio_compressor_get_configuration(struct pressio_compressor const* compressor) {
  return compressor->plugin->get_configuration();
}
struct pressio_options* pressio_compressor_get_options(struct pressio_compressor const* compressor) {
  return compressor->plugin->get_options();
}
int pressio_compressor_set_options(struct pressio_compressor* compressor, struct pressio_options const * options) {
  return compressor->plugin->set_options(options);
}
int pressio_compressor_compress(struct pressio_compressor* compressor, const pressio_data *input, struct pressio_data * output) {
  return compressor->plugin->compress(input, output);
}
int pressio_compressor_decompress(struct pressio_compressor* compressor, const pressio_data *input, struct pressio_data * output) {
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

struct pressio_options* pressio_compressor_get_metrics_results(struct pressio_compressor const* compressor) {
  return compressor->plugin->get_metrics_results();
}

struct pressio_metrics* pressio_compressor_get_metrics(struct pressio_compressor const* compressor) {
  return compressor->plugin->get_metrics();
}

void pressio_compressor_set_metrics(struct pressio_compressor* compressor, struct pressio_metrics* plugin) {
  return compressor->plugin->set_metrics(plugin);
}

void pressio_compressor_release(struct pressio_compressor* compressor) {
  delete compressor;
}

}
