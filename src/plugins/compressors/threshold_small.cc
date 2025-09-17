#include <iostream>
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"

namespace libpressio { namespace compressors { namespace threshold_small_ns {

class threshold_small_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "threshold_small:compressor", compressor_id, compressor);
    set(options, "threshold_small:threshold", threshold);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    std::vector<std::string> invalidations {"threshold_small:threshold"}; 
    std::vector<pressio_configurable const*> invalidation_children {&*compressor}; 
    set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
    set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
    set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));
    set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{}));
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_docs(options, "threshold_small:compressor", "the compressor after applying the threshold", compressor);
    set(options, "pressio:description", R"(applies a treshold to small values making them 0)");
    set(options, "threshold_small:threshold", R"(threshold to apply)");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "threshold_small:threshold", &threshold);
    get_meta(options, "threshold_small:compressor", compressor_plugins(), compressor_id, compressor);
    return 0;
  }

  struct apply_threshold {
      template <class T>
      pressio_data operator()(T* begin, T* end) {
          size_t N = end-begin;
          pressio_data out = pressio_data::clone(input);
          T* in_ptr = static_cast<T*>(input.data());
          T* out_ptr = static_cast<T*>(out.data());
          for(size_t i= 0; i < N; ++i) {
              if(in_ptr[i] > threshold) {
                out_ptr[i] = in_ptr[i];
              } else {
                out_ptr[i] = 0;
              }
          }
          return out;
      }
      pressio_data const& input;
      double threshold;
  };

  int compress_impl(const pressio_data* real_input,
                    struct pressio_data* output) override
  {
      pressio_data input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
      pressio_data thresholded = pressio_data_for_each<pressio_data>(input, apply_threshold{input, threshold});
      int rc = compressor->compress(&thresholded, output);
      if(rc) {
          return set_error(compressor->error_code(), compressor->error_msg());
      }
      return 0;
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    
      int rc = compressor->decompress(input, output);
      if(rc) {
          return set_error(compressor->error_code(), compressor->error_msg());
      }
      return 0;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "threshold_small"; }

  void set_name_impl(std::string const& new_name) override {
      compressor->set_name(new_name + "/thresholded");
  }
  pressio_options get_metrics_results_impl() const override {
    return compressor->get_metrics_results();
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<threshold_small_compressor_plugin>(*this);
  }

  double threshold=0;
  std::string compressor_id = "noop";
  pressio_compressor compressor = compressor_plugins().build(compressor_id);
};

pressio_register registration(compressor_plugins(), "threshold_small", []() {
  return compat::make_unique<threshold_small_compressor_plugin>();
});

} } }

