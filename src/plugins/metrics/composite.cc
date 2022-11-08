#include <vector>
#include <memory>
#include <functional>
#include "libpressio_ext/cpp/pressio.h"
#include "pressio_options.h"
#include "pressio_compressor.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "std_compat/memory.h"
#include "std_compat/string_view.h"
#include "pressio_version.h"
#if LIBPRESSIO_HAS_LUA
#define SOL_ALL_SAFETIES_ON 1
#define SOL_PRINT_ERRORS 1
#include <sol/sol.hpp>
#endif

namespace libpressio { namespace composite {
class composite_plugin : public libpressio_metrics_plugin {
  public:
  explicit composite_plugin(std::vector<pressio_metrics>&& plugins) :
    plugins(std::move(plugins))
    {
      std::transform(std::begin(this->plugins),
          std::end(this->plugins),
          std::back_inserter(names),
          std::mem_fn(&libpressio_metrics_plugin::prefix)
          );
      std::transform(std::begin(this->plugins),
          std::end(this->plugins),
          std::back_inserter(plugins_ids),
          std::mem_fn(&libpressio_metrics_plugin::prefix)
          );
    }

  composite_plugin():
    composite_plugin(std::vector<pressio_metrics>{}) {}
  int begin_check_options_impl(struct pressio_options const* options) override {
    for (auto& plugin : plugins) {
      plugin->begin_check_options(options);
    }
    return 0;
  }

  int end_check_options_impl(struct pressio_options const* options, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_check_options(options, rc);
    }
    return 0;
  }

  int begin_get_options_impl() override {
    for (auto& plugin : plugins) {
      plugin->begin_get_options();
    }
    return 0;
  }

  int end_get_options_impl(struct pressio_options const* options) override {
    for (auto& plugin : plugins) {
      plugin->end_get_options(options);
    }
    return 0;
  }

  int begin_set_options_impl(struct pressio_options const& options) override {
    for (auto& plugin : plugins) {
      plugin->begin_set_options(options);
    }
    return 0;
  }

  int end_set_options_impl(struct pressio_options const& options, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_set_options(options, rc);
    }
    return 0;
  }

  int begin_compress_impl(const struct pressio_data * input, struct pressio_data const * output) override {
    for (auto& plugin : plugins) {
      plugin->begin_compress(input, output);
    }
    return 0;
  }

  int end_compress_impl(struct pressio_data const* input, pressio_data const * output, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_compress(input, output, rc);
    }
    return 0;
  }

  int begin_decompress_impl(struct pressio_data const* input, pressio_data const* output) override {
    for (auto& plugin : plugins) {
      plugin->begin_decompress(input, output);
    }
    return 0;
  }

  int end_decompress_impl(struct pressio_data const* input, pressio_data const* output, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_decompress(input, output, rc);
    }
    return 0;
  }

  int begin_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs) override {
    for (auto& plugin : plugins) {
      plugin->begin_compress_many(inputs, outputs);
    }
    return 0;
  }

  int end_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_compress_many(inputs, outputs, rc);
    }

    return 0;
  }

  int begin_decompress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs) override {
    for (auto& plugin : plugins) {
      plugin->begin_decompress_many(inputs, outputs);
    }
    return 0;
  }

  int end_decompress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_decompress_many(inputs, outputs, rc);
    }
    return 0;
 
  }

  pressio_options get_metrics_results(pressio_options const &)  override {
    struct pressio_options metrics_result;
    for (auto const& plugin : plugins) {
      pressio_options plugin_options = plugin->get_metrics_results({});
      auto tmp = pressio_options_merge(&metrics_result, &plugin_options);
      metrics_result = std::move(*tmp);
      pressio_options_free(tmp);
    }
    set_composite_metrics(metrics_result);

    return metrics_result;
  }

  struct pressio_options get_options() const override {
    struct pressio_options metrics_options;
    set_meta_many(metrics_options, "composite:plugins", plugins_ids, plugins);
    set(metrics_options, "composite:names", names);
#if LIBPRESSIO_HAS_LUA
    set(metrics_options, "composite:scripts", scripts);
#endif
    return metrics_options;
  }

  int set_options(pressio_options const& options) override {
    int rc = 0;
    get(options, "composite:names", &names);
    get_meta_many(options, "composite:plugins", metrics_plugins(), plugins_ids, plugins);
    for (auto const& plugin : plugins) {
      rc |= plugin->set_options(options);
    }
#if LIBPRESSIO_HAS_LUA
    get(options, "composite:scripts", &scripts);
#endif
    return rc;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<composite_plugin>(*this);
  }

  const char* prefix() const override {
    return "composite";
  }

  struct pressio_options get_configuration() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    return opts;
  }


  pressio_options get_documentation_impl() const override {
    pressio_options options;
    set_meta_many_docs(options, "composite:plugins",  "plugins used for gathering metrics", plugins);
    set(options, "composite:compression_rate", "compression rate for the compress method, activated by size and time");
    set(options, "composite:compression_rate_many", "compression rate for the compress_many method, activated by size and time");
    set(options, "composite:decompression_rate", "decompression rate for the compress method, activated by size and time");
    set(options, "composite:decompression_rate_many", "decompression rate for the compress_many method, activated by size and time");
    set(options, "composite:names", "the names to use for the constructed metrics plugins");
    set(options, "composite:scripts", "a lua script used to compute metrics from other metrics that have been previously computed");
    set(options, "pressio:description", "meta-metric that runs a set of metrics in sequence");

    return options;
  }

  protected:
  void set_name_impl(std::string const& name) override {
    set_names_many(name, plugins, names);
  };

  private:

  int set_composite_metrics(struct pressio_options& opt) {
    std::string time_name;
    std::string size_name;
    for (const auto & plugin : plugins) {
      if(compat::string_view(plugin->prefix()) == "time") {
        time_name = plugin->get_name();
      } else if(compat::string_view(plugin->prefix()) == "size") {
        size_name = plugin->get_name();
      }
    }

    //compression_rate
    uint64_t uncompressed_size;
    {
      unsigned int compression_time;
      if(opt.get(time_name ,"time:compress", &compression_time) == pressio_options_key_set &&
         opt.get(size_name ,"size:uncompressed_size", &uncompressed_size) == pressio_options_key_set) {
        set(opt, "composite:compression_rate", static_cast<double>(uncompressed_size)/compression_time);
      } else {
        set_type(opt, "composite:compression_rate", pressio_option_double_type);
      }
    }

    //decompression_rate
    {
      unsigned int decompression_time;
      if (opt.get(time_name,  "time:decompress", &decompression_time) == pressio_options_key_set &&
          opt.get(size_name, "size:uncompressed_size", &uncompressed_size) == pressio_options_key_set) {
        set(opt, "composite:decompression_rate", static_cast<double>(uncompressed_size)/decompression_time);
      } else {
        set_type(opt, "composite:decompression_rate", pressio_option_double_type);
      }
    }
    
    //compression_rate_many
    {
      unsigned int compression_time;
      if(opt.get(time_name ,"time:compress_many", &compression_time) == pressio_options_key_set &&
         opt.get(size_name ,"size:uncompressed_size", &uncompressed_size) == pressio_options_key_set) {
        set(opt, "composite:compression_rate_many", static_cast<double>(uncompressed_size)/compression_time);
      } else {
        set_type(opt, "composite:compression_rate_many", pressio_option_double_type);
      }
    }

    //decompression_rate_many
    {
      unsigned int decompression_time;
      if (opt.get(time_name,  "time:decompress_many", &decompression_time) == pressio_options_key_set &&
          opt.get(size_name, "size:uncompressed_size", &uncompressed_size) == pressio_options_key_set) {
        set(opt, "composite:decompression_rate_many", static_cast<double>(uncompressed_size)/decompression_time);
      } else {
        set_type(opt, "composite:decompression_rate_many", pressio_option_double_type);
      }
    }


#if LIBPRESSIO_HAS_LUA
    std::map<std::string, double> metrics;
    for (auto const& o: opt) {
      auto o_as_double = o.second.as(pressio_option_double_type, pressio_conversion_explicit);
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

        sol::optional<std::tuple<std::string, sol::optional<double>>> lua_result = lua.safe_script(script, 
            [this](lua_State*, sol::protected_function_result pfr) {
                sol::error err = pfr;
                set_error(1, std::string("lua error: ") + err.what());
                return pfr;
        });
        if(lua_result) {
          auto const& lua_result_v = *lua_result;
          std::string name = std::string("composite:") + std::get<0>(lua_result_v);
          if(std::get<1>(lua_result_v)) {
            set(opt, name, *std::get<1>(lua_result_v));
            metrics[name] = *std::get<1>(lua_result_v);
          } else {
            set_type(opt, name, pressio_option_double_type);
          }
        } else {
          return error_code();
        }
      } catch (sol::error& err) {
        return set_error(1, std::string("lua error; ") + err.what());
      }
    }
#endif

    return 0;

  }

  std::vector<pressio_metrics> plugins;
  std::vector<std::string> names;
  std::vector<std::string> plugins_ids;
#if LIBPRESSIO_HAS_LUA
  std::vector<std::string> scripts;
#endif
};

static pressio_register metrics_composite_plugin(metrics_plugins(), "composite", [](){ return compat::make_unique<composite_plugin>(); });


} }

std::unique_ptr<libpressio_metrics_plugin> make_m_composite(std::vector<pressio_metrics>&& plugins) {
  return compat::make_unique<libpressio::composite::composite_plugin>(std::move(plugins));
}
