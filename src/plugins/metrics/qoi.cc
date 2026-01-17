// #include "pressio_options.h"
// #include "libpressio_ext/cpp/metrics.h"
// #include "libpressio_ext/cpp/pressio.h"
// #include "std_compat/memory.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/data.h"
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
    // Iterate over all keys in child's results to find JSON-parsed keys
    // JSON format: {"mean": 158.7, "std": 2.5} -> "external:results:mean", "external:results:std"
    // Key-value format: "data=158.7\ndata=159.2..." -> "external:results:data" (pressio_data array)
    const std::string prefix = "external:results:";
    std::vector<double> values;  // Collect all values from JSON keys
    bool found_any_json_key = false;
    // Step 1: Iterate through all keys in child's results to find JSON keys
    fprintf(stderr, "[QOI] DEBUG: Searching for JSON keys with prefix '%s':\n", prefix.c_str());
    for (auto const& item : opt) {
      const std::string& key = item.first;
      
      // Check if key starts with "external:results:" (JSON-parsed keys)
      if (key.find(prefix) == 0) {
        std::string json_key = key.substr(prefix.length());  // Extract JSON key (e.g., "mean")
        fprintf(stderr, "[QOI] Found JSON key: '%s' -> full key: '%s'\n", json_key.c_str(), key.c_str());
        found_any_json_key = true;
        
        // Try to get as pressio_data (for arrays)
        if (get(opt, key, &qoi_data) == pressio_options_key_set) {
          fprintf(stderr, "[QOI]   - Key '%s' is pressio_data array\n", json_key.c_str());
          auto* ptr = static_cast<const double*>(qoi_data.data());
          size_t n = qoi_data.num_elements();
          

        }
        // Try to get as double (for single values)
        else if (get(opt, key, &qoi_value) == pressio_options_key_set) {
          fprintf(stderr, "[QOI]   - Key '%s' is double: %f\n", json_key.c_str(), qoi_value);
          values.push_back(qoi_value);
        }
      }
    }

    
    // Step 3: Convert collected values to pressio_data and iterate with pointer
    if (found_any_json_key && !values.empty()) {
      // Create pressio_data from collected values using copy (which copies the data)
      pressio_data qoi_result = pressio_data::copy(pressio_double_dtype, values.data(), {values.size()});
      
      // Iterate through the data using pointer
      const double* ptr = static_cast<const double*>(qoi_result.data());
      size_t n = qoi_result.num_elements();
      
      fprintf(stderr, "[QOI] Created pressio_data with %zu elements, iterating with pointer:\n", n);
      // for (size_t i = 0; i < n; ++i) {
        // fprintf(stderr, "[QOI]   [%zu] = %f\n", i, ptr[i]);
        
        // Example: if this is the first value and it's "mean", set qoi:mean
        // if (i == 0) {
        //   // Assuming first value is mean (can be customized based on your needs)
        //   set(opt, "qoi:mean", ptr[i]);
        //   fprintf(stderr, "[QOI] Set qoi:mean=%f from pressio_data pointer[0]\n", ptr[i]);
        // }
      // }
      
      // Store the entire pressio_data as qoi:data
      set(opt, "qoi:data", qoi_result);
    } else {
      // Step 4: If still not found, report warning
      fprintf(stderr, "[QOI] WARNING: No JSON keys found with prefix '%s' in child results or parent\n", prefix.c_str());
      fprintf(stderr, "[QOI] Available keys in child results:\n");
      for (auto const& item : opt) {
        fprintf(stderr, "[QOI]   - %s\n", item.first.c_str());
      }
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

  std::string qoi = "external:results";  // Base prefix for JSON results: {"mean": value} -> external:results:mean
  pressio_metrics child = metrics_plugins().build("noop");
  std::string child_id = "noop";
  // double mean = 0.0;
};

pressio_register registration(metrics_plugins(), "qoi", [](){   std::cout << "Registering qoi plugin\n"; return compat::make_unique<qoi_plugin>(); });

}}}