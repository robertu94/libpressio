#include <set>
#include <string>
#include <algorithm>
#include <iterator>
#include <sstream>
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"

#include "libpressio_ext/cpp/pressio.h"
#include "pressio_options_iter.h"
#include "pressio_options.h"

libpressio_compressor_plugin::libpressio_compressor_plugin() noexcept :
  pressio_configurable(),
  metrics_plugin(metrics_plugins().build("noop")),
  metrics_id("noop")
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
  clear_error();

  if(metrics_plugin)
    metrics_plugin->begin_check_options(&options);

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
  if(metrics_plugin)
    metrics_plugin->end_check_options(&options, ret);

  return ret;
}

struct pressio_options libpressio_compressor_plugin::get_configuration() const {
  pressio_options ret;
  if(metrics_plugin){
    set_meta_configuration(ret, "pressio:metric", metrics_plugins(), metrics_plugin);
    set_meta_configuration(ret, get_metrics_key_name(), metrics_plugins(), metrics_plugin);
    metrics_plugin->begin_get_configuration();
  }
  ret.copy_from(get_configuration_impl());
  set(ret, "pressio:version_epoch", epoch_version());
  set(ret, "pressio:version_major", major_version());
  set(ret, "pressio:version_minor", minor_version());
  set(ret, "pressio:version_patch", patch_version());
  set(ret, "pressio:version", version());
  if(metrics_plugin) { 
    metrics_plugin->end_get_configuration(ret);
  }
  return ret;
}

struct pressio_options libpressio_compressor_plugin::get_documentation() const {
  if(metrics_plugin)
    metrics_plugin->begin_get_documentation();
  pressio_options ret;
  if(metrics_plugin) { 
    set_meta_docs(ret, "pressio:metric", "metrics to collect when using the compressor", metrics_plugin);
    set_meta_docs(ret, get_metrics_key_name(), "metrics to collect when using the compressor", metrics_plugin);
  }
  set(ret, "pressio:thread_safe", R"(level of thread safety provided by the compressor

  pressio_thread_safety_single = 0, indicates not thread safe
  pressio_thread_safety_serialized = 1, indicates individual handles may be called from different threads sequentially
  pressio_thread_safety_multiple = 2, indicates individual handles may be called from different threads concurrently
  )");
  set(ret, "pressio:stability", R"(level of stablity provided by the compressor

  + experimental: Modules that are experimental may crash or have other severe deficiencies,
  + unstable: modules that are unstable generally will not crash, but may have options changed according to the unstable API guarantees.
  + stable: conforms to the LibPressio stability guarantees
  + external: indicates that options/configuration returned by this module are controlled by version of the external library that it depends upon and may change at any time without changing the LibPressio version number.
  )");
  set(ret, "pressio:version_epoch", R"(the epoch version number; this is a libpressio specific value used if the major_version does not accurately reflect backward incompatibility)");
  set(ret, "pressio:version_major", R"(the major version number)");
  set(ret, "pressio:version_minor", R"(the minor version number)");
  set(ret, "pressio:version_patch", R"(the patch version number)");
  set(ret, "pressio:version", R"(the version string from the compressor)");

  set(ret, "pressio:nthreads", R"(number of threads to use)");
  set(ret, "pressio:abs", R"(a pointwise absolute error bound

  compressors may provide this value without supporting abs=0.
  compressors that support abs=0, additionally should also define pressio:lossless
  )");
  set(ret, "pressio:rel", R"(a pointwise value-range relative error bound

  compressors may provide this value without supporting rel=0.
  compressors that support rel=0, additionally should also define pressio:lossless
  )");
  set(ret, "pressio:pw_rel", R"(a pointwise relative error bound

  compressors may provide this value without supporting pw_rel=0.
  compressors that support pw_rel=0, additionally should also define pressio:lossless
  )");
  set(ret, "pressio:lossless", R"(use lossless compression,
    the smaller the number the more biased towards speed,
    the larger the number the more biased towards compression

    at this time (may change in the future), individual lossless compressors may internet values less than 1 or greater than 9 differently
  )");
  set(ret, "pressio:lossless:min", R"(minimum compression level for pressio:lossless)");
  set(ret, "pressio:lossless:max", R"(maximum compression level for pressio:lossless)");
  ret.copy_from(get_documentation_impl());
  if(metrics_plugin) { 
    metrics_plugin->end_get_documentation(ret);
  }
  return ret;
}

struct pressio_options libpressio_compressor_plugin::get_options() const {
  if(metrics_plugin)
    metrics_plugin->begin_get_options();
  pressio_options opts;
  set_meta(opts, "pressio:metric", metrics_id, metrics_plugin);
  set_meta(opts, get_metrics_key_name(), metrics_id, metrics_plugin);
  set(opts, "metrics:errors_fatal", metrics_errors_fatal);
  set(opts, "metrics:copy_compressor_results", metrics_copy_impl_results);
  opts.copy_from(get_options_impl());
  if(metrics_plugin)
    metrics_plugin->end_get_options(&opts);
  return opts;
}

int libpressio_compressor_plugin::set_options(struct pressio_options const& options) {
  clear_error();
  if(metrics_plugin) {
    if(metrics_plugin->begin_set_options(options) != 0 && metrics_errors_fatal) {
      set_error(metrics_plugin->error_code(), metrics_plugin->error_msg());
      return error_code();
    }
  }
  get_meta(options, "pressio:metric", metrics_plugins(), metrics_id, metrics_plugin);
  get_meta(options, get_metrics_key_name(), metrics_plugins(), metrics_id, metrics_plugin);
  get(options, "metrics:errors_fatal", &metrics_errors_fatal);
  get(options, "metrics:copy_compressor_results", &metrics_copy_impl_results);
  auto ret = set_options_impl(options);
  if(metrics_plugin) {
    if(metrics_plugin->end_set_options(options, ret) != 0 && metrics_errors_fatal) {
      set_error(metrics_plugin->error_code(), metrics_plugin->error_msg());
      return error_code();
    }
  }
  return ret;
}

int libpressio_compressor_plugin::compress(const pressio_data *input, struct pressio_data* output) {
  clear_error();
  if(metrics_plugin) {
    if(metrics_plugin->begin_compress(input, output) != 0 && metrics_errors_fatal) {
      set_error(metrics_plugin->error_code(), metrics_plugin->error_msg());
      return error_code();
    }
  }
  auto ret = compress_impl(input, output);
  if(metrics_plugin) {
    if(metrics_plugin->end_compress(input, output, ret) != 0 && metrics_errors_fatal) {
      set_error(metrics_plugin->error_code(), metrics_plugin->error_msg());
      return error_code();
    }
  }
  return ret;
}

int libpressio_compressor_plugin::decompress(const pressio_data *input, struct pressio_data* output) {
  clear_error();
  if(metrics_plugin)
    metrics_plugin->begin_decompress(input, output);
  auto ret = decompress_impl(input, output);
  if(metrics_plugin)
    metrics_plugin->end_decompress(input, output, ret);
  return ret;
}

int libpressio_compressor_plugin::check_options_impl(struct pressio_options const &) { return 0;}


struct pressio_options libpressio_compressor_plugin::get_metrics_results() const {
  pressio_options results_impl = get_metrics_results_impl();
  pressio_options results;
  if(metrics_copy_impl_results) {
    results.copy_from(results_impl);
  }
  if(metrics_plugin) {
    results.copy_from(metrics_plugin->get_metrics_results(results_impl));
  }
  return results;
}

struct pressio_metrics libpressio_compressor_plugin::get_metrics() const& {
  return metrics_plugin;
}

struct pressio_metrics&& libpressio_compressor_plugin::get_metrics() && {
  return std::move(metrics_plugin);
}

void libpressio_compressor_plugin::set_metrics(pressio_metrics& plugin) {
  metrics_plugin = plugin;
  if(plugin) {
    metrics_id = metrics_plugin->prefix();
    if(not get_name().empty()) {
      metrics_plugin->set_name(get_name() + "/" + metrics_plugin->prefix());
    }
  } else {
    metrics_id = "";
  }
}

void libpressio_compressor_plugin::set_metrics(pressio_metrics&& plugin) {
  metrics_plugin = std::move(plugin);
  if(metrics_plugin) {
    metrics_id = metrics_plugin->prefix();
    if(not get_name().empty()) {
      metrics_plugin->set_name(get_name() + "/" + metrics_plugin->prefix());
    }
  } else {
    metrics_id = "";
  }
}


struct pressio_options libpressio_compressor_plugin::get_metrics_options() const {
  return metrics_plugin->get_options();
}

int libpressio_compressor_plugin::set_metrics_options(struct pressio_options const& options) {
  clear_error();
  return metrics_plugin->set_options(options);
}

struct pressio_options libpressio_compressor_plugin::get_metrics_results_impl() const {
  return {};
}

int libpressio_compressor_plugin::compress_many_impl(compat::span<const pressio_data* const> const& inputs, compat::span<pressio_data*> & outputs) {
    //default returns an error to indicate the option is unsupported;
    if(inputs.size() == 1 && outputs.size() == 1) {
      return compress_impl(inputs.front(), outputs.front());
    } else 
    return set_error(1, "decompress_many not supported");
  }

int libpressio_compressor_plugin::decompress_many_impl(compat::span<const pressio_data* const> const& inputs, compat::span<pressio_data* >& outputs) {
    //default returns an error to indicate the option is unsupported;
    if(inputs.size() == 1 && outputs.size() == 1) {
      return decompress_impl(inputs.front(), outputs.front());
    } else 
    return set_error(1, "decompress_many not supported");
  }


void libpressio_compressor_plugin::set_name(std::string const& new_name) {
  if (new_name != "") {
    pressio_configurable::set_name(new_name);
    metrics_plugin->set_name(new_name + "/" + metrics_plugin->prefix());
  } else {
    pressio_configurable::set_name(new_name);
    metrics_plugin->set_name(new_name);
  }
}
