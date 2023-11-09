#include "libpressio_ext/cpp/configurable.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"

libpressio_metrics_plugin::libpressio_metrics_plugin():
  pressio_configurable()
{}

struct pressio_options libpressio_metrics_plugin::get_documentation() const {
  auto opts = get_documentation_impl();
  set(opts, "pressio:thread_safe", "level of thread safety provided by the compressor");
  set(opts, "pressio:stability", "level of stablity provided by the compressor; see the README for libpressio");
  set(opts, "metrics:copy_compressor_results", "copy the metrics provided by the compressor");
  set(opts, "metrics:errors_fatal", "propagate errors from the metrics to the compressor");
  set(opts, "pressio:type", R"(type of the libpressio meta object)");
  set(opts, "pressio:children", R"(children of this libpressio meta object)");
  set(opts, "pressio:prefix", R"(prefix of the this libpresiso meta object)");
  return opts;
}

struct pressio_options libpressio_metrics_plugin::get_configuration_impl() const {
    return {};
}

struct pressio_options libpressio_metrics_plugin::get_configuration() const {
  auto opts = get_configuration_impl();
  set(opts, "pressio:children", children());
  set(opts, "pressio:type", type());
  set(opts, "pressio:prefix", prefix());
  return opts;
}

int libpressio_metrics_plugin::begin_check_options(struct pressio_options const * opts) {
  clear_error();
  return begin_check_options_impl(opts);
}
int libpressio_metrics_plugin::end_check_options(struct pressio_options const * opts, int rc) {
  clear_error();
  return end_check_options_impl(opts, rc);
}
int libpressio_metrics_plugin::begin_get_documentation() {
  clear_error();
  return begin_get_documentation_impl();
}
int libpressio_metrics_plugin::end_get_documentation(struct pressio_options const & opts) {
  clear_error();
  return end_get_documentation_impl(opts);
}
int libpressio_metrics_plugin::begin_get_configuration() {
  clear_error();
  return begin_get_configuration_impl();
}
int libpressio_metrics_plugin::end_get_configuration(struct pressio_options const & opts) {
  clear_error();
  return end_get_configuration_impl(opts);
}
int libpressio_metrics_plugin::begin_get_options() {
  clear_error();
  return begin_get_options_impl();
}
int libpressio_metrics_plugin::end_get_options(struct pressio_options const * opts) {
  clear_error();
  return end_get_options_impl(opts);
}
int libpressio_metrics_plugin::begin_set_options(struct pressio_options const & opts) {
  clear_error();
  return begin_set_options_impl(opts);
}
int libpressio_metrics_plugin::end_set_options(struct pressio_options const & opts, int rc) {
  clear_error();
  return end_set_options_impl(opts, rc);
}
int libpressio_metrics_plugin::begin_compress(const struct pressio_data * input, struct pressio_data const * output) {
  clear_error();
  return begin_compress_impl(input, output);
}
int libpressio_metrics_plugin::end_compress(struct pressio_data const * input, pressio_data const * output, int rc) {
  clear_error();
  return end_compress_impl(input, output, rc);
}
int libpressio_metrics_plugin::begin_decompress(struct pressio_data const * input, pressio_data const * output) {
  clear_error();
  return begin_decompress_impl(input, output);
}
int libpressio_metrics_plugin::end_decompress(struct pressio_data const * input, pressio_data const * output, int rc) {
  clear_error();
  return end_decompress_impl(input, output, rc);
}
int libpressio_metrics_plugin::begin_compress_many(compat::span<const pressio_data* const> const& inputs,
                                                        compat::span<const pressio_data* const> const& outputs) {
  clear_error();
  return begin_compress_many_impl(inputs, outputs);
}
int libpressio_metrics_plugin::end_compress_many(compat::span<const pressio_data* const> const& inputs,
                                                      compat::span<const pressio_data* const> const& outputs, int rc) {
  clear_error();
  return end_compress_many_impl(inputs, outputs, rc);
}
int libpressio_metrics_plugin::begin_decompress_many(compat::span<const pressio_data* const> const& inputs,
                                                          compat::span<const pressio_data* const> const& outputs) {
  clear_error();
  return begin_decompress_many_impl(inputs, outputs);
}
int libpressio_metrics_plugin::end_decompress_many(compat::span<const pressio_data* const> const& inputs,
                                                        compat::span<const pressio_data* const> const& outputs, int rc) {
  clear_error();
  return end_decompress_many_impl(inputs, outputs, rc);
}
int libpressio_metrics_plugin::begin_check_options_impl(struct pressio_options const *) {
  return 0;
}
int libpressio_metrics_plugin::end_check_options_impl(struct pressio_options const *, int ) {
  return 0;
}
int libpressio_metrics_plugin::begin_get_documentation_impl() {
  return 0;
}
int libpressio_metrics_plugin::end_get_documentation_impl(struct pressio_options const &) {
  return 0;
}
int libpressio_metrics_plugin::begin_get_configuration_impl() {
  return 0;
}
int libpressio_metrics_plugin::end_get_configuration_impl(struct pressio_options const &) {
  return 0;
}
int libpressio_metrics_plugin::begin_get_options_impl() {
  return 0;
}
int libpressio_metrics_plugin::end_get_options_impl(struct pressio_options const *) {
  return 0;
}
int libpressio_metrics_plugin::begin_set_options_impl(struct pressio_options const &) {
  return 0;
}
int libpressio_metrics_plugin::end_set_options_impl(struct pressio_options const &, int ) {
  return 0;
}
int libpressio_metrics_plugin::begin_compress_impl(const struct pressio_data *, struct pressio_data const *) {
  return 0;
}
int libpressio_metrics_plugin::end_compress_impl(struct pressio_data const *, pressio_data const *, int ) {
  return 0;
}
int libpressio_metrics_plugin::begin_decompress_impl(struct pressio_data const *, pressio_data const *) {
  return 0;
}
int libpressio_metrics_plugin::end_decompress_impl(struct pressio_data const *, pressio_data const *, int) {
  return 0;
}
int libpressio_metrics_plugin::begin_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                 compat::span<const pressio_data* const> const& outputs) {
  if(inputs.size() == 1 && outputs.size() == 1) {
    return begin_compress_impl(inputs.front(), outputs.front());
  }
  return 0;
}
int libpressio_metrics_plugin::end_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                 compat::span<const pressio_data* const> const& outputs, int rc) {
  if(inputs.size() == 1 && outputs.size() == 1) {
    return end_compress_impl(inputs.front(), outputs.front(), rc);
  }
  return 0;
}
int libpressio_metrics_plugin::begin_decompress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                 compat::span<const pressio_data* const> const& outputs) {
  if(inputs.size() == 1 && outputs.size() == 1) {
    return begin_decompress_impl(inputs.front(), outputs.front());
  }
  return 0;
}
int libpressio_metrics_plugin::end_decompress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                 compat::span<const pressio_data* const> const& outputs, int rc) {
  if(inputs.size() == 1 && outputs.size() == 1) {
    return end_decompress_impl(inputs.front(), outputs.front(), rc);
  }
  return 0;
}

int libpressio_metrics_plugin::view_segment(pressio_data const* data, const char* segment_id) {
    return view_segment_impl(data, segment_id);
}

int libpressio_metrics_plugin::view_segment_impl(pressio_data const*, const char*) {
    return 0;
}

void libpressio_metrics_plugin::set_name(std::string const& new_name) {
  pressio_configurable::set_name(new_name);
}
