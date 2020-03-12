#include <set>
#include <string>
#include <algorithm>
#include <iterator>
#include <sstream>
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"

#include "pressio_options_iter.h"
#include "pressio_options.h"

libpressio_compressor_plugin::libpressio_compressor_plugin() noexcept :
  pressio_configurable(),
  pressio_errorable(),
  metrics_plugin(nullptr)
{}

libpressio_compressor_plugin::~libpressio_compressor_plugin()=default;


namespace {
  std::set<std::string> get_keys(struct pressio_options const& options, std::string const& prefix) {
    std::set<std::string> keys;
    for (auto const& option : options) {
      if(option.first.find(prefix) == 0)
        keys.emplace(option.first);
    }
    return keys;
  }
}

int libpressio_compressor_plugin::check_options(struct pressio_options const& options) {

  if(metrics_plugin) metrics_plugin->begin_check_options(&options);

  struct pressio_options my_options = get_options();
  auto my_keys = get_keys(my_options, prefix());
  auto keys = get_keys(options, prefix());
  std::set<std::string> extra_keys;
  std::set_difference(
      std::begin(keys), std::end(keys),
      std::begin(my_keys), std::end(my_keys),
      std::inserter(extra_keys, std::begin(extra_keys))
  );
  if(!extra_keys.empty()) {
    std::stringstream ss;
    ss << "extra keys: ";

    std::copy(std::begin(extra_keys), std::end(extra_keys), std::ostream_iterator<std::string>(ss, " "));
    set_error(1, ss.str());
    return 1;
  }

  auto ret =  check_options_impl(options);
  if(metrics_plugin) metrics_plugin->end_check_options(&options, ret);

  return ret;
}

struct pressio_options libpressio_compressor_plugin::get_configuration() const {
  if(metrics_plugin) metrics_plugin->begin_get_configuration();
  auto ret = get_configuration_impl();
  if(metrics_plugin) metrics_plugin->end_get_configuration(ret);
  return ret;
}

struct pressio_options libpressio_compressor_plugin::get_options() const {
  if(metrics_plugin) metrics_plugin->begin_get_options();
  auto ret = get_options_impl();
  if(metrics_plugin) metrics_plugin->end_get_options(&ret);
  return ret;
}

int libpressio_compressor_plugin::set_options(struct pressio_options const& options) {
  if(metrics_plugin) metrics_plugin->begin_set_options(options);
  auto ret = set_options_impl(options);
  if(metrics_plugin) metrics_plugin->end_set_options(options, ret);
  return ret;
}

int libpressio_compressor_plugin::compress(const pressio_data *input, struct pressio_data* output) {
  if(metrics_plugin) metrics_plugin->begin_compress(input, output);
  auto ret = compress_impl(input, output);
  if(metrics_plugin) metrics_plugin->end_compress(input, output, ret);
  return ret;
}

int libpressio_compressor_plugin::decompress(const pressio_data *input, struct pressio_data* output) {
  if(metrics_plugin) metrics_plugin->begin_decompress(input, output);
  auto ret = decompress_impl(input, output);
  if(metrics_plugin) metrics_plugin->end_decompress(input, output, ret);
  return ret;
}

int libpressio_compressor_plugin::check_options_impl(struct pressio_options const &) { return 0;}


struct pressio_options libpressio_compressor_plugin::get_metrics_results() const {
  return metrics_plugin->get_metrics_results();
}

struct pressio_metrics libpressio_compressor_plugin::get_metrics() const {
  return metrics_plugin;
}

void libpressio_compressor_plugin::set_metrics(pressio_metrics& plugin) {
  metrics_plugin = plugin;
}

struct pressio_options libpressio_compressor_plugin::get_metrics_options() const {
  return metrics_plugin->get_options();
}

int libpressio_compressor_plugin::set_metrics_options(struct pressio_options const& options) {
  return metrics_plugin->set_options(options);
}

