#include "libpressio_ext/cpp/configurable.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"

void libpressio_metrics_plugin::begin_check_options(struct pressio_options const*) {
}
void libpressio_metrics_plugin::end_check_options(struct pressio_options const*, int) {
}
void libpressio_metrics_plugin::begin_get_configuration() {
}
void libpressio_metrics_plugin::end_get_configuration(struct pressio_options const&) {
}
void libpressio_metrics_plugin::begin_get_options() {
}
void libpressio_metrics_plugin::end_get_options(struct pressio_options const*) {
}
void libpressio_metrics_plugin::begin_set_options(struct pressio_options const&) {
}
void libpressio_metrics_plugin::end_set_options(struct pressio_options const&, int) {
}
void libpressio_metrics_plugin::begin_compress(const struct pressio_data *, struct pressio_data const *) {
}
void libpressio_metrics_plugin::end_compress(struct pressio_data const*, pressio_data const *, int) {
}
void libpressio_metrics_plugin::begin_decompress(struct pressio_data const*, pressio_data const*) {
}
void libpressio_metrics_plugin::end_decompress(struct pressio_data const*, pressio_data const*, int) {
}
void libpressio_metrics_plugin::begin_compress_many(compat::span<const pressio_data* const> const& inputs,
                                 compat::span<const pressio_data* const> const& outputs) {
  if(inputs.size() == 1 && outputs.size() == 1) {
    begin_compress(inputs.front(), outputs.front());
  }
}
void libpressio_metrics_plugin::end_compress_many(compat::span<const pressio_data* const> const& inputs,
                                 compat::span<const pressio_data* const> const& outputs, int rc) {
  if(inputs.size() == 1 && outputs.size() == 1) {
    end_compress(inputs.front(), outputs.front(), rc);
  }
}
void libpressio_metrics_plugin::begin_decompress_many(compat::span<const pressio_data* const> const& inputs,
                                 compat::span<const pressio_data* const> const& outputs) {
  if(inputs.size() == 1 && outputs.size() == 1) {
    begin_decompress(inputs.front(), outputs.front());
  }
}
void libpressio_metrics_plugin::end_decompress_many(compat::span<const pressio_data* const> const& inputs,
                                 compat::span<const pressio_data* const> const& outputs, int rc) {
  if(inputs.size() == 1 && outputs.size() == 1) {
    end_decompress(inputs.front(), outputs.front(), rc);
  }
}

void libpressio_metrics_plugin::set_name(std::string const& new_name) {
  pressio_configurable::set_name(new_name);
}
