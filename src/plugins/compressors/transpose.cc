#include <vector>
#include <memory>
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "pressio_options.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "std_compat/memory.h"

namespace libpressio { namespace transpose {

class transpose_meta_compressor_plugin : public libpressio_compressor_plugin
{
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options = compressor->get_options();
    set_meta(options, "transpose:compressor", compressor_id, compressor);
    set(options, "transpose:axis", pressio_data(axis.begin(), axis.end()));
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set_meta_configuration(options, "transpose:compressor", compressor_plugins(), compressor);
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "unstable");
    set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", {&*compressor}, {"transpose:axis"}));
    set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", {&*compressor}, {}));
    set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", {&*compressor}, {}));
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_docs(options, "transpose:compressor", "Compressor to use after transpose is applied", compressor);
    set(options, "pressio:description", "Meta-compressor that applies a transpose before compression");
    set(options, "transpose:axis", "how to reorder the dimensions, contains indicies 0..(N_DIMS-1)");
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override
  {
    get_meta(options, "transpose:compressor", compressor_plugins(), compressor_id, compressor);
    pressio_data tmp;
    if(get(options, "transpose:axis", &tmp) == pressio_options_key_set) {
      axis = tmp.to_vector<size_t>();
    }
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    auto tmp = domain_manager().make_readable(domain_plugins().build("malloc"),*input).transpose(axis);
    return compressor->compress(&tmp, output);
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    auto ret = compressor->decompress(input, output);
    *output = domain_manager().make_readable(domain_plugins().build("malloc"), std::move(*output));
    output->transpose(axis);
    return ret;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }

  const char* version() const override { return "0.0.1"; }

  const char* prefix() const override { return "transpose"; }

  void set_name_impl(std::string const& name) override {
    if(name != "") {
      compressor->set_name(name + "/" + compressor->prefix());
    } else {
      compressor->set_name(name);
    }
  }
  std::vector<std::string> children_impl() const final {
      return { compressor->get_name() };
  }

  pressio_options get_metrics_results_impl() const override {
    return compressor->get_metrics_results();
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<transpose_meta_compressor_plugin>(*this);
  }

private:
  std::vector<size_t> axis;
  pressio_compressor compressor = compressor_plugins().build("noop");
  std::string compressor_id = "noop";
};

static pressio_register compressor_transpose_plugin(compressor_plugins(), "transpose", [](){ return compat::make_unique<transpose_meta_compressor_plugin>(); });


} } 
