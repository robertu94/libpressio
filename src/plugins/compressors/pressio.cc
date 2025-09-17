#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"

namespace libpressio { namespace compressors { namespace pressio_ns {

  enum class pressio_mode {
    abs,        /*absolute error bound emulation*/
    rel,        /*value-range relative emulation*/
    pw_rel,     /*point-wise relative emulation*/
    passthrough,/*do nothing to modify the settings*/
    forward,    /*when pressio:bound is set, set pressio:bound_name; useful for giving a
                  consistent name to a setting for applications like z-checker*/
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
    set(options, "pressio:bound_name", bound_name);
    if(bound_name) {
      auto status = child.key_status(*bound_name);
      if(status == pressio_options_key_set || status == pressio_options_key_exists) {
        set(options, "pressio:bound", child.get(bound_name));
      }
    }
    set_type(options, "pressio:reset_mode", pressio_option_bool_type);

    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set_meta_configuration(options, "pressio:compressor", compressor_plugins(), comp);
    set(options, "pressio:thread_safe", get_threadsafe(*comp));
    set(options, "pressio:stability", "experimental");
    
        std::vector<std::string> invalidations {"pressio:bound_name", "pressio:reset_mode"}; 
        auto child = comp->get_options();
        if(child_supports_mode(child, "pressio:abs")) {
            invalidations.emplace_back("pressio:abs");
        }
        if(child_supports_mode(child, "pressio:rel")) {
            invalidations.emplace_back("pressio:rel");
        }
        std::vector<pressio_configurable const*> invalidation_children {&*comp}; 
        
        set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
        set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
        set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, {}));

    
        set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, invalidations));

    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_docs(options, "pressio:compressor", "the underlying compressor to use", comp);
    set(options, "pressio:description", R"(a set of helpers to convert between common error bounds)");
    set(options, "pressio:reset_mode", "reset mode back to none");
    set(options, "pressio:bound", "forward the bound that is provided to pressio:bound_name");
    set(options, "pressio:bound_name", "passthrough the bound that is provided");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "pressio:bound_name", &bound_name);
    get_meta(options, "pressio:compressor", compressor_plugins(), comp_id, comp);
    if(bound_name && options.key_status(get_name(), "pressio:bound") == pressio_options_key_set) {
      pressio_options const& child_options = comp->get_options();
      auto status = child_options.key_status(*bound_name);
      if(status == pressio_options_key_set || status == pressio_options_key_exists) {
        pressio_option const& child_option = child_options.get(*bound_name);
        pressio_option const& bound = options.get("pressio:bound");

        pressio_options new_options;
        new_options.set(*bound_name, child_option);
        new_options.cast_set(*bound_name, bound, pressio_conversion_special);

        comp->set_options(new_options);
        mode= pressio_mode::forward;
      } else {
        return set_error(1, "option does not exist: " + *bound_name);
      }
    }
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

  int compress_impl(const pressio_data* real_input,
                    struct pressio_data* output) override
  {
    auto child = comp->get_options();
    //default to non-owning, and then move if needed
    pressio_data input = pressio_data::nonowning(*real_input);
    switch(mode) {
      case pressio_mode::passthrough:
      case pressio_mode::forward:
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
          input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
          //TODO add fallback for devices to avoid unnecessary copies
          double value_range = pressio_data_for_each<double>(*real_input, compute_value_range{});
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

    if(comp->compress(&input, output)) {
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
  std::vector<std::string> children_impl() const final {
      return {comp->get_name()};
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
  compat::optional<std::string> bound_name;
  std::string comp_id = "noop";
  pressio_compressor comp = compressor_plugins().build("noop");
};

pressio_register registration(compressor_plugins(), "pressio", []() {
  return compat::make_unique<pressio_compressor_plugin>();
});

} }}
