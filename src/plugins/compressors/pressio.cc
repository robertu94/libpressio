#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"

namespace libpressio { namespace pressio_ns {

  enum class pressio_mode {
    abs,
    rel,
    pw_rel,
    passthrough
  };

  struct compute_value_range {
    template <class T>
    double operator()(T const* begin, T const* end) {
      auto minmax = std::minmax_element(begin, end);
      return *minmax.second - *minmax.first;
    }
  };

class pressio_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "pressio:compressor", comp_id, comp);

    auto child = comp->get_options();
    if(child_supports_mode(child, "pressio:abs")) {
      if(mode == pressio_mode::abs) {
        set(options, "pressio:abs", target);
      } else {
        set_type(options, "pressio:abs", pressio_option_double_type);
      }
      if(mode == pressio_mode::rel) {
        set(options, "pressio:rel", target);
      } else {
        set_type(options, "pressio:rel", pressio_option_double_type);
      }
    }
    set_type(options, "pressio:reset_mode", pressio_option_bool_type);

    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    options.copy_from(comp->get_configuration());
    set(options, "pressio:thread_safe", static_cast<int32_t>(get_threadsafe(*comp)));
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_docs(options, "pressio:compressor", "the underlying compressor to use", comp);
    set(options, "pressio:description", R"(a set of helpers to convert between common error bounds)");
    set(options, "pressio:reset_mode", "reset mode back to none");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get_meta(options, "pressio:compressor", compressor_plugins(), comp_id, comp);
    bool reset_mode = false;
    get(options, "pressio:reset_mode", &reset_mode);
    if(reset_mode) {
      mode = pressio_mode::passthrough;
    }

    auto child = comp->get_options();
    if(child_supports_mode(child, "pressio:abs")) {
      if(get(options, "pressio:abs", &target) == pressio_options_key_set) {
        mode = pressio_mode::abs;
      }
      if(get(options, "pressio:rel", &target) == pressio_options_key_set) {
        mode = pressio_mode::rel;
      }
    }

    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    auto child = comp->get_options();
    switch(mode) {
      case pressio_mode::passthrough:
        //don't set modify configuration in this mode
        break;
      case pressio_mode::abs:
        if(child_supports_mode(child, "pressio:abs")) {
          //should already be set by inheritance, ignore
        } else {
          return set_error(2, "pressio:abs is not supported for " + comp_id);
        }
        break;
      case pressio_mode::rel:
        if(child_supports_mode(child, "pressio:rel")){
          //should already be set by inheritance, ignore
        } else if(child_supports_mode(child, "pressio:abs")) {
          pressio_options options;
          double value_range = pressio_data_for_each<double>(*input, compute_value_range{});
          set(options, "pressio:abs", target*value_range);
          comp->set_options(options);
        }
        break;
      case pressio_mode::pw_rel:
        if(child_supports_mode(child, "pressio:pw_rel")) {
          //should already be supported by inheritance, ignore
        } else {
          return set_error(2, "pressio:pw_rel is not supported for " + comp_id);
        }
        break;
      default:
        return set_error(1, "unknown mode mode");
    }

    if(comp->compress(input, output)) {
      return set_error(comp->error_code(), comp->error_msg());
    }
    return 0;
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    if(comp->decompress(input, output)) {
      return set_error(comp->error_code(), comp->error_msg());
    }
    return 0;
  }

  void set_name_impl(std::string const& name) override {
    if(name != "") {
    comp->set_name(name + '/' + comp->prefix());
    } else {
    comp->set_name(name);
    }
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "pressio"; }

  pressio_options get_metrics_results_impl() const override {
    return comp->get_metrics_results();
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<pressio_compressor_plugin>(*this);
  }

private:
  bool child_supports_mode(pressio_options const& child, const char* mode_str) const {
      return child.key_status(comp->get_name(), mode_str) <= pressio_options_key_exists;
  }

  pressio_mode mode = pressio_mode::passthrough;
  double target;
  std::string comp_id = "noop";
  pressio_compressor comp = compressor_plugins().build("noop");
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "pressio", []() {
  return compat::make_unique<pressio_compressor_plugin>();
});

} }
