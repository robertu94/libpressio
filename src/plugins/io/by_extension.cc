#include <map>
#include <set>
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/io.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "std_compat/memory.h"

namespace libpressio { namespace by_extension_io_ns {

class by_extension_plugin : public libpressio_io_plugin {
  public:
  virtual struct pressio_data* read_impl(struct pressio_data* buf) override {
    auto ret = plugin->read(buf);
    if(ret == nullptr) {
      set_error(plugin->error_code(), plugin->error_msg());
    }
    return ret;
  }
  virtual int write_impl(struct pressio_data const* data) override{
    auto ret =  plugin->write(data);
    if(ret) {
      set_error(plugin->error_code(), plugin->error_msg());
    }
    return ret;
  }

  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "by_extension:plugin", plugin_id, plugin);
    std::vector<std::string> keys, values;
    for (auto const& i : mapping) {
      keys.emplace_back(i.first);
      keys.emplace_back(i.second);
    }
    set_type(options, "io:path", pressio_option_charptr_type);
    set(options, "by_extension:keys", keys);
    set(options, "by_extension:values", values);

    return options;
  }

  int set_options_impl(struct pressio_options const& options) override
  {
    std::vector<std::string> keys, values;
    pressio_options_key_status keys_found = get(options, "by_extension:keys", &keys);
    pressio_options_key_status values_found = get(options, "by_extension:values", &values);
    if(keys_found == pressio_options_key_set && values_found == pressio_options_key_set) {
      if(keys.size() != values.size()) return set_error(3, "invalid mapping: num keys != num values");
      if (std::find(keys.begin(), keys.end(), "") != keys.end()) {
        return set_error(3,
                         "invalid mapping: a fallback is required, provide an empty string as key");
      }
      std::set<std::string> test_unique(keys.begin(), keys.end());
      if(test_unique.size() != keys.size()) {
        return set_error(4,
                         "invalid mapping: a duplicate key was provided");
      }
      mapping.clear();
      for (size_t i = 0; i < values.size(); ++i) {
        mapping[keys[i]] = values[i];
      }
    }

    std::string path;
    if(get(options, "io:path", &path)==pressio_options_key_set && path.find('.') != std::string::npos) {
      auto extension = path.substr(path.rfind('.')+1);
      auto try_set_plugin = [this](const std::string& plugin_type){
        if(io_plugins().find(plugin_type) != io_plugins().end()) {
          plugin_id = plugin_type;
          plugin = io_plugins().build(plugin_id);
        } else {
          set_error(1, std::string("required plugin ") + plugin_id);
        }
      };
      try {
        try_set_plugin(mapping.at(extension));
      } catch(std::out_of_range&) {
        try_set_plugin(mapping.at(""));
      }
    }
    get_meta(options, "by_extension:plugin", io_plugins(), plugin_id, plugin);
    return error_code();
  }

  const char* version() const override { return "0.0.1"; }
  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }

  
  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    opts.copy_from(plugin->get_configuration());
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set_meta_docs(opt, "by_extension:plugin", "io plugin to delegate to", plugin);
    set(opt, "pressio:description", "a \"by_extension\" data loader that chooses an appropriate io plugin based on file extension");
    set(opt, "by_extension:keys", "extensions to map");
    set(opt, "by_extension:values", "io plugins to map to");
    return opt;
  }
  void set_name_impl(std::string const& new_name) override {
    plugin->set_name(new_name + "/" + plugin->prefix());
  }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<by_extension_plugin>(*this);
  }
  const char* prefix() const override {
    return "by_extension";
  }

  private:
  std::map<std::string, std::string> mapping {
      {"npy", "numpy"},
      {"h5", "hdf5"},
      {"hdf5", "hdf5"},
      {"csv", "csv"},
      {"bp", "adios2"},
      {"nc", "netcdf"},
      { "", "posix"}
  };
  std::string plugin_id = "posix";
  pressio_io plugin = io_plugins().build(plugin_id);
};

static pressio_register io_by_extension_plugin(io_plugins(), "by_extension", [](){ return compat::make_unique<by_extension_plugin>(); });
}}
