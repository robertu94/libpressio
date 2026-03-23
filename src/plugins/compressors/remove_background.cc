#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"

namespace libpressio { namespace compressors { namespace remove_background_ns {

class remove_background_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "remove_background:compressor", compressor_id, compressor);
    set(options, "remove_background:background", background);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    std::vector<std::string> invalidations {"remove_background:background"}; 
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
    set_meta_docs(options, "remove_background:compressor", "the compressor after removing the background", compressor);
    set(options, "pressio:description", R"(removes the background from an image and then compresses)");
    set(options, "remove_background:background", R"(background to remove)");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "remove_background:background", &background);
    get_meta(options, "remove_background:compressor", compressor_plugins(), compressor_id, compressor);
    return 0;
  }
  struct remove_background{
      template <class T>
      pressio_data operator()(T* begin, T* end) {
          size_t N = end-begin;
          pressio_data out = pressio_data::clone(input);
          T* background_ptr = static_cast<T*>(background.data());
          T* out_ptr = static_cast<T*>(out.data());
          for(size_t i= 0; i < N; ++i) {
            out_ptr[i] -= background_ptr[i];
          }
          return out;
      }
      pressio_data const& input;
      pressio_data const& background;
  };

  int compress_impl(const pressio_data* real_input,
                    struct pressio_data* output) override
  {
      if(real_input->dimensions() != background.dimensions()) {
          return set_error(1, "input and background dimensions do not match");
      }
      pressio_data input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
      pressio_data background_removed = pressio_data_for_each<pressio_data>(input, remove_background{input, background});
      int rc = compressor->compress(&background_removed, output);
      if(rc) {
          return set_error(compressor->error_code(), compressor->error_msg());
      }
      return 0;
  }

  struct restore_background{
      template <class T>
      int operator()(T* begin, T* end) {
          size_t N = end-begin;
          T* background_ptr = static_cast<T*>(background.data());
          T* out_ptr = static_cast<T*>(out.data());
          for(size_t i= 0; i < N; ++i) {
            out_ptr[i] -= background_ptr[i];
          }
          return 0;
      }
      pressio_data & out;
      pressio_data const& background;
  };

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
      if(output->dimensions() != background.dimensions()) {
          return set_error(1, "output and background dimensions do not match");
      }
      int rc = compressor->decompress(input, output);
      if(rc) {
          return set_error(compressor->error_code(), compressor->error_msg());
      }
      *output = domain_manager().make_readable(domain_plugins().build("malloc"), std::move(*output));
      pressio_data_for_each<int>(*output, restore_background{*output, background});
      return 0;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "remove_background"; }

  void set_name_impl(std::string const& new_name) override {
      compressor->set_name(new_name + "/background");
  }
  pressio_options get_metrics_results_impl() const override {
    return compressor->get_metrics_results();
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<remove_background_compressor_plugin>(*this);
  }

  pressio_data background;
  std::string compressor_id = "noop";
  pressio_compressor compressor = compressor_plugins().build(compressor_id);
};

pressio_register registration(compressor_plugins(), "remove_background", []() {
  return compat::make_unique<remove_background_compressor_plugin>();
});

} }
}
