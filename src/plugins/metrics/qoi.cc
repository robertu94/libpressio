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
#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>




namespace libpressio { namespace metrics {

namespace qoi_ns {

class qoi_plugin : public libpressio_metrics_plugin {
  public:
  int set_options(pressio_options const& options) override {
    fprintf(stderr, "[QOI] set_options called\n");
    fflush(stderr);
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
    printf("--------------");
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

  pressio_options get_metrics_results(pressio_options const & parent) override {
    pressio_options opt = child->get_metrics_results(parent);
    pressio_data qoi_data;
    double qoi_value;

    // Step 5: Get pressio_data from parent (when used with composite, parent contains accumulated results)
    // First try to get as pressio_data
    if(get(parent, qoi, &qoi_data) == pressio_options_key_set) {
      fprintf(stderr, "[QOI] Got pressio_data from parent, key=%s\n", qoi.c_str());
      // Step 5: Access values inside pressio_data
      // Use templated accessors to safely iterate over the data
      auto* ptr = static_cast<const double*>(qoi_data.data());
      size_t n = qoi_data.num_elements();
      fprintf(stderr, "[QOI] pressio_data: n=%zu, dtype=%d\n", n, qoi_data.dtype());
      
      // Calculate mean from pressio_data
      if (n > 0 && qoi_data.dtype() == pressio_double_dtype) {
        double mean = std::accumulate(ptr, ptr + n, 0.0) / static_cast<double>(n);
        fprintf(stderr, "[QOI] Calculated mean=%f from pressio_data\n", mean);
        set(opt, "qoi:mean", mean);
        
        // Verify it was set
        double verify_mean;
        if(get(opt, "qoi:mean", &verify_mean) == pressio_options_key_set) {
          fprintf(stderr, "[QOI] Verified qoi:mean=%f is set in opt\n", verify_mean);
        } else {
          fprintf(stderr, "[QOI] ERROR: qoi:mean was NOT set in opt!\n");
        }
      }
    }
    // Fallback: if pressio_data not found, try to get as double
    else if(get(parent, qoi, &qoi_value) == pressio_options_key_set) {
      fprintf(stderr, "[QOI] Got double from parent, key=%s, value=%f\n", qoi.c_str(), qoi_value);
      // For single double value, mean is the value itself
      set(opt, "qoi:mean", qoi_value);
      
      // Verify it was set
      double verify_mean;
      if(get(opt, "qoi:mean", &verify_mean) == pressio_options_key_set) {
        fprintf(stderr, "[QOI] Verified qoi:mean=%f is set in opt\n", verify_mean);
      } else {
        fprintf(stderr, "[QOI] ERROR: qoi:mean was NOT set in opt!\n");
      }
    } else {
      fprintf(stderr, "[QOI] WARNING: Could not get %s from parent\n", qoi.c_str());
    }

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