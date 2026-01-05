// #include "pressio_options.h"
// #include "libpressio_ext/cpp/metrics.h"
// #include "libpressio_ext/cpp/pressio.h"
// #include "std_compat/memory.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"


// #include <nlohmann/json.hpp>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>




namespace libpressio { namespace metrics {

namespace qoi_ns {

class qoi_plugin : public libpressio_metrics_plugin {
  public:
  int set_options(pressio_options const& options) override {
    get_meta(options, "qoi:metric", metrics_plugins(), child_id, child);
    // options.get("qoi:metric_name", &metric_name);
    return 0;
  }
  pressio_options get_options() const override {
    pressio_options opts;
    set_meta(opts, "qoi:metric", child_id, child);
    // opts.set("qoi:metric_name", metric_name);
    return opts;
  }

  int begin_check_options_impl(struct pressio_options const* opts) override {
    return child->begin_check_options(opts);
  }

  int end_check_options_impl(struct pressio_options const* opts, int rc) override {
    return child->end_check_options(opts, rc);
  }

  int begin_get_options_impl() override {
    return child->begin_get_options();
  }

  int end_get_options_impl(struct pressio_options const* opts) override {
    return child->end_get_options(opts);
  }

  int begin_get_configuration_impl() override {
    return child->begin_get_configuration();
  }

  int end_get_configuration_impl(struct pressio_options const& opts) override {
    return child->end_get_configuration(opts);
  }

  int begin_set_options_impl(struct pressio_options const& opts) override {
    return child->begin_set_options(opts);
  }

  int end_set_options_impl(struct pressio_options const& opts, int rc) override {
    return child->end_set_options(opts, rc);
  }

  int begin_compress_impl(const struct pressio_data * input, struct pressio_data const * output) override {
    return child->begin_compress(input, output);
  }

  int end_compress_impl(struct pressio_data const* input, pressio_data const * output, int rc) override {
    // Logic to calculate min_v, max_v, p99_v, p999_v, wasserstein_v would go here
    return child->end_compress(input, output, rc);
  }

  int begin_decompress_impl(struct pressio_data const* input, pressio_data const* output) override {
    return child->begin_decompress(input, output);
  }

  int end_decompress_impl(struct pressio_data const* input, pressio_data const* output, int rc) override {
    return child->end_decompress(input, output, rc);
  }

  int begin_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs) override {
    return child->begin_compress_many(inputs, outputs);
  }

  int end_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs, int rc) override {
    return child->end_compress_many(inputs, outputs, rc);
  }

  int begin_decompress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs) override {
    return child->begin_decompress_many(inputs, outputs);
  }

  int end_decompress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs, int rc) override {
    return child->end_decompress_many(inputs, outputs, rc);
  }

  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set_meta_configuration(opts, "qoi:metric", metrics_plugins(), child);
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(opts, "predictors:requires_decompress", std::vector<std::string>{"time:decompress", "time:decompress_many", "time:begin_decompress", "time:end_decompress", "time:begin_decompress_many", "time:end_decompress_many"});
    set(opts, "predictors:invalidate", std::vector<std::string>{"predictors:runtime", "predictors:nondeterministc"});
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", R"()");
    return opt;

  }

  pressio_options get_metrics_results(pressio_options const & parent)  override {

    pressio_options opt;
    opt.copy_from(child->get_metrics_results(opt));
    pressio_data qoi_data;

    // if(get(opt, qoi, &qoi_data) == pressio_options_key_set) {
    //   printf("success------------------------------------");

    // }



    return opt;
  
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<qoi_plugin>(*this);
  }

  const char* prefix() const override {
    return "qoi";
  }



  private:

  std::string qoi = "external:results:data";
  pressio_metrics child = metrics_plugins().build("noop");
  std::string child_id = "noop";
  // double mean = 0.0;
};

pressio_register registration(metrics_plugins(), "qoi", [](){   std::cout << "Registering qoi plugin\n"; return compat::make_unique<qoi_plugin>(); });

}}}
