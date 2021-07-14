#include <libpressio_ext/cpp/compressor.h>
#include <libpressio_ext/cpp/pressio.h>
#include <std_compat/memory.h>
#include <sstream>

class switch_compressor: public libpressio_compressor_plugin {
  pressio_options get_options_impl() const override {
    pressio_options opts;
    set_meta_many(opts, "switch:compressors", compressor_ids, compressors);
    set(opts, "switch:active_id", active_id);
    set_type(opts, "switch:clear_invocations", pressio_option_int32_type);
    return opts;
  }
  pressio_options get_documentation_impl() const override {
    pressio_options opts;
    set_meta_many_docs(opts, "switch:compressors", "compressor ids to configure", compressors);
    set(opts, "pressio:description", R"(switch_compressor

    Allows switching between different compressors at runtime
    )");
    set(opts, "switch:clear_invocations", "*write-only* clear the invocation count metric");
    set(opts, "switch:active_id", "the compressor to actually use");
    return opts;
  }
  pressio_options get_configuration_impl() const override {
    pressio_options opts;
    try {
      set(opts, "pressio:thread_safe", static_cast<int32_t>( pressio_configurable::get_threadsafe(*compressors.at(active_id))));
    } catch(std::out_of_range const&) {
      set(opts, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_single));
    }
    set(opts, "pressio:stability", "experimental");
    return opts;
  }
  int set_options_impl(const pressio_options &options) override {
    get_meta_many(options, "switch:compressors", compressor_plugins(), compressor_ids, compressors);
    get(options, "switch:active_id", &active_id);
    int32_t clear = 0;
    get(options, "switch:clear_invocations", &clear);
    if(clear) {
      compression_invocations.clear();
    }
    compression_invocations.resize(compressors.size());

    return 0;
  }
  int compress_impl(const pressio_data *input, struct pressio_data *output) override {
    try {
    compression_invocations.at(active_id)++;
    return compressors.at(active_id)->compress(input, output);
    } catch(std::out_of_range& ex) {
      return set_error(1, std::string("invalid active_id: ") + ex.what());
    }
  }
  int decompress_impl(const pressio_data *input, struct pressio_data *output) override {
    try {
    return compressors.at(active_id)->decompress(input, output);
    } catch(std::out_of_range& ex) {
      return set_error(1, std::string("invalid active_id: ") + ex.what());
    }
  }
  void set_name_impl(std::string const& name) override {
    set_names_many(name, compressors, names);
  }

  pressio_options get_metrics_results_impl() const override {
    pressio_options opts;
    for (auto const& plugin : compressors) {
      opts.copy_from(plugin->get_metrics_results());
    }
    set(opts, "switch:compression_invocations", pressio_data(compression_invocations.begin(), compression_invocations.end()));
    return opts;
  }

  const char* prefix() const override { return "switch"; }
  const char* version() const override { 
    const static std::string version_str = [this]{
      std::stringstream ss;
      ss << major_version() << '.' << minor_version() << '.' << patch_version();
      return ss.str();
    }();
    return version_str.c_str();
  }
  virtual std::shared_ptr<libpressio_compressor_plugin> clone() override {
    return compat::make_unique<switch_compressor>(*this);
  }

  uint64_t active_id = 0;
  std::vector<std::string> names;
  std::vector<std::string> compressor_ids;
  std::vector<pressio_compressor> compressors;
  std::vector<uint64_t> compression_invocations;
};

static pressio_register switch_compressor_register(
    compressor_plugins(),
    "switch",
    []{
      return compat::make_unique<switch_compressor>();
    }
    );
