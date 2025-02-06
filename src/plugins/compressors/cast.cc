
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"

namespace libpressio { namespace cast_ns {

class cast_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "cast:compressor", compressor_id, compressor);
    set(options, "cast:dtype", compress_dtype);
    set(options, "cast:decompress_dtype", decompress_dtype);
    set(options, "cast:restore_type", restore_dtype);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    std::vector<std::string> invalidations {}; 
    std::vector<pressio_configurable const*> invalidation_children {}; 
    set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
    set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
    set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));
    set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{}));
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_docs(options, "cast:compressor", "change the datatype of the input", compressor);
    set(options, "pressio:description", R"(cast inputs to different datatypes before compression/decompression)");
    set(options, "cast:dtype", R"(cast to on copmress)");
    set(options, "cast:decompress_dtype", R"(cast to on copmress decompress if restore_type is not true)");
    set(options, "cast:restore_type", R"(set the decompress dtype on call to compress)");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "cast:dtype", &compress_dtype);
    get(options, "cast:decompress_dtype", &decompress_dtype);
    get(options, "cast:restore_type", &restore_dtype);
    get_meta(options, "cast:compressor", compressor_plugins(), compressor_id, compressor);
    return 0;
  }

  int compress_impl(const pressio_data* real_input,
                    struct pressio_data* output) override
  {
      pressio_data input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
      if(restore_dtype) decompress_dtype = input.dtype();
      pressio_data casted = input.cast(compress_dtype);
      int rc = compressor->compress(&casted, output);
      if(rc) {
          return set_error(compressor->error_code(), compressor->error_msg());
      }
      return 0;
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* real_output) override
  {
      //TODO real_output's dtype is probably wrong here. it should be the casted compress_dtype
      int rc = compressor->decompress(input, real_output);
      if(rc) {
          return set_error(compressor->error_code(), compressor->error_msg());
      }
      pressio_data output = domain_manager().make_readable(domain_plugins().build("malloc"), *real_output);
      *real_output = output.cast(decompress_dtype);
      return 0;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "cast"; }

  void set_name_impl(std::string const& new_name) override {
      if(!new_name.empty()) compressor->set_name(new_name + "/casted");
      else compressor->set_name("");
  }
  pressio_options get_metrics_results_impl() const override {
    return compressor->get_metrics_results();
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<cast_compressor_plugin>(*this);
  }

  std::string compressor_id = "noop";
  pressio_compressor compressor = compressor_plugins().build(compressor_id);
  pressio_dtype compress_dtype = pressio_float_dtype;
  pressio_dtype decompress_dtype = pressio_float_dtype;
  bool restore_dtype = true;

};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "cast", []() {
  return compat::make_unique<cast_compressor_plugin>();
});

} }

