#include <vector>
#include <memory>
#include <functional>
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/compat/memory.h"
#if LIBPRESSIO_HAS_LUA
#define SOL_ALL_SAFETIES_ON 1
#define SOL_PRINT_ERRORS 1
#include <sol/sol.hpp>
#endif

using namespace std::literals;

class composite_plugin : public libpressio_metrics_plugin {
  public:
  composite_plugin(std::vector<std::unique_ptr<libpressio_metrics_plugin>>&& plugins) :
    plugins(std::move(plugins))
    {
      std::transform(std::begin(plugins),
          std::end(plugins),
          std::back_inserter(names),
          std::mem_fn(&libpressio_metrics_plugin::prefix)
          );
    }
  void begin_check_options(struct pressio_options const* options) override {
    for (auto& plugin : plugins) {
      plugin->begin_check_options(options);
    }
  }

  void end_check_options(struct pressio_options const* options, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_check_options(options, rc);
    }
  }

  void begin_get_options() override {
    for (auto& plugin : plugins) {
      plugin->begin_get_options();
    }
  }

  void end_get_options(struct pressio_options const* options) override {
    for (auto& plugin : plugins) {
      plugin->end_get_options(options);
    }
  }

  void begin_set_options(struct pressio_options const& options) override {
    for (auto& plugin : plugins) {
      plugin->begin_set_options(options);
    }
  }

  void end_set_options(struct pressio_options const& options, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_set_options(options, rc);
    }
  }

  void begin_compress(const struct pressio_data * input, struct pressio_data const * output) override {
    for (auto& plugin : plugins) {
      plugin->begin_compress(input, output);
    }
  }

  void end_compress(struct pressio_data const* input, pressio_data const * output, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_compress(input, output, rc);
    }
  }

  void begin_decompress(struct pressio_data const* input, pressio_data const* output) override {
    for (auto& plugin : plugins) {
      plugin->begin_decompress(input, output);
    }
  }

  void end_decompress(struct pressio_data const* input, pressio_data const* output, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_decompress(input, output, rc);
    }
  }

  void begin_compress_many(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs) override {
    for (auto& plugin : plugins) {
      plugin->begin_compress_many(inputs, outputs);
    }
  }

  void end_compress_many(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_compress_many(inputs, outputs, rc);
    }
   
  }

  void begin_decompress_many(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs) override {
    for (auto& plugin : plugins) {
      plugin->begin_decompress_many(inputs, outputs);
    }
  }

  void end_decompress_many(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_decompress_many(inputs, outputs, rc);
    }
 
  }

  struct pressio_options get_metrics_results() const override {
    struct pressio_options metrics_result;
    for (auto const& plugin : plugins) {
      pressio_options plugin_options = plugin->get_metrics_results();
      auto tmp = pressio_options_merge(&metrics_result, &plugin_options);
      metrics_result = std::move(*tmp);
      pressio_options_free(tmp);
    }
    set_composite_metrics(metrics_result);

    return metrics_result;
  }

  struct pressio_options get_options() const override {
    struct pressio_options metrics_options;
    for (auto const& plugin : plugins) {
      pressio_options plugin_options = plugin->get_options();
      auto tmp = pressio_options_merge(&metrics_options, &plugin_options);
      metrics_options = std::move(*tmp);
      pressio_options_free(tmp);
    }
    set(metrics_options, "composite:names", names);
#if LIBPRESSIO_HAS_LUA
    set(metrics_options, "composite:scripts", scripts);
#endif
    return metrics_options;
  }

  int set_options(pressio_options const& options) override {
    int rc = 0;
    for (auto const& plugin : plugins) {
      rc |= plugin->set_options(options);
    }
    get(options, "composite:names", &names);
#if LIBPRESSIO_HAS_LUA
    get(options, "composite:scripts", &scripts);
#endif
    return rc;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    std::vector<std::unique_ptr<libpressio_metrics_plugin>> cloned;
    for (auto& plugin : plugins) {
      cloned.emplace_back(plugin->clone());
    }
    return compat::make_unique<composite_plugin>(std::move(cloned));
  }

  const char* prefix() const override {
    return "composite";
  }

  protected:
  void set_name_impl(std::string const& name) override {
    for (size_t i = 0; i < plugins.size(); ++i) {
      plugins[i]->set_name(name + "/" + names[i]);
    }
  };

  private:

  void set_composite_metrics(struct pressio_options& opt) const {
    std::string time_name;
    std::string size_name;
    for (size_t i = 0; i < plugins.size(); ++i) {
      if(plugins[i]->prefix() == "time"s) {
        time_name = plugins[i]->get_name();
      } else if(plugins[i]->prefix() == "size"s) {
        size_name = plugins[i]->get_name();
      }
    }

    //compression_rate
    {
      unsigned int compression_time, uncompressed_size;
      if(opt.get(time_name ,"time:compress", &compression_time) == pressio_options_key_set &&
         opt.get(size_name ,"size:uncompressed_size", &uncompressed_size) == pressio_options_key_set) {
        set(opt, "composite:compression_rate", static_cast<double>(uncompressed_size)/compression_time);
      } else {
        set_type(opt, "composite:compression_rate", pressio_option_double_type);
      }
    }

    //decompression_rate
    {
      unsigned int decompression_time, uncompressed_size;
      if (opt.get(time_name,  "time:decompress", &decompression_time) == pressio_options_key_set &&
          opt.get(size_name, "size:uncompressed_size", &uncompressed_size) == pressio_options_key_set) {
        set(opt, "composite:decompression_rate", static_cast<double>(uncompressed_size)/decompression_time);
      } else {
        set_type(opt, "composite:decompression_rate", pressio_option_double_type);
      }
    }

#if LIBPRESSIO_HAS_LUA
    std::map<std::string, double> metrics;
    for (auto const& o: opt) {
      auto o_as_double = o.second.as(pressio_option_double_type, pressio_conversion_implicit);
      if(o_as_double.has_value()) {
        metrics[o.first] = o_as_double.get_value<double>();
      }
    }
    for (auto const& script : scripts) {
      try {
        //create a new state for each object to ensure it is clean
        sol::state lua;
        lua.open_libraries(sol::lib::base);
        lua.open_libraries(sol::lib::math);
        lua["metrics"] = metrics;

        sol::optional<std::tuple<std::string, sol::optional<double>>> lua_result = lua.safe_script(script);
        if(lua_result) {
          auto const& lua_result_v = *lua_result;
          std::string name = std::string("composite:") + std::get<0>(lua_result_v);
          if(std::get<1>(lua_result_v)) {
            set(opt, name, *std::get<1>(lua_result_v));
            metrics[name] = *std::get<1>(lua_result_v);
          } else {
            set_type(opt, name, pressio_option_double_type);
          }
        }
      } catch (sol::error& err) {
        //swallow errors from sol and do not insert a key
      }
    }
#endif

  }

  std::vector<std::unique_ptr<libpressio_metrics_plugin>> plugins;
  std::vector<std::string> names;
#if LIBPRESSIO_HAS_LUA
  std::vector<std::string> scripts;
#endif
};

std::unique_ptr<libpressio_metrics_plugin> make_m_composite(std::vector<std::unique_ptr<libpressio_metrics_plugin>>&& plugins) {
  return compat::make_unique<composite_plugin>(std::move(plugins));
}
