/**
 * a dummy no-op compressor for use in testing and facilitating querying parameters
 */
#include <memory>

#include "pressio_compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "std_compat/memory.h"

namespace libpressio { namespace noop_compressor {

class noop_compressor_plugin: public libpressio_compressor_plugin {
  public:

  struct pressio_options get_configuration_impl() const override {
    pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "stable");
    
        std::vector<std::string> invalidations {}; 
        std::vector<pressio_configurable const*> invalidation_children {}; 
        
        set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
        set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
        set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));

    return options;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options options;
    set(options, "pressio:description", "A no-op compressor useful for testing and defaults");
    return options;
  }

  struct pressio_options get_options_impl() const override {
    return {};
  }

  int set_options_impl(struct pressio_options const&) override {
    return 0;
  }

  int compress_impl(const pressio_data *input, struct pressio_data* output) override {
    *output = pressio_data::clone(*input);
    return 0;
  }

  int compress_many_impl(compat::span<const pressio_data *const> const& input, compat::span<pressio_data*>& output) override {
    for (size_t i = 0; i < std::min(input.size(), output.size()); ++i) {
      *output[i] = pressio_data::clone(*input[i]);
    }
    return 0;
  }

  int decompress_impl(const pressio_data *input, struct pressio_data* output) override {
		*output = pressio_data::copy(output->dtype(), input->data(), output->dimensions());
    return 0;
  }

  int decompress_many_impl(compat::span<const pressio_data *const> const& input, compat::span<pressio_data*>& output) override {
    for (size_t i = 0; i < std::min(input.size(), output.size()); ++i) {
      *output[i] = pressio_data::copy(output[i]->dtype(), input[i]->data(), output[i]->dimensions());
    }
    return 0;
  }


  int major_version() const override {
    return 0;
  }
  int minor_version() const override {
    return 0;
  }
  int patch_version() const override {
    return 0;
  }

  const char* version() const override {
    return "noop 0.0.0.0"; 
  }

  const char* prefix() const override {
    return "noop";
  }
  std::shared_ptr<libpressio_compressor_plugin> clone() override{
    return compat::make_unique<noop_compressor_plugin>(*this);
  }
};

static pressio_register comprssor_noop_plugin(compressor_plugins(), "noop", [](){ return compat::make_unique<noop_compressor_plugin>();});

} }
