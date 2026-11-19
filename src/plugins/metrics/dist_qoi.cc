#include "pressio_options.h"
#include "pressio_version.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/data.h"
#include "std_compat/memory.h"

#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cmath>
#include <limits>


namespace libpressio { namespace metrics {

namespace dist_qoi_ns {


struct dist_qoi_statistics {
  double mean;
  double min_val;
  double max_val;
  double median;
  double p90;
  double p99;
  double p999;
  double wasserstein_distance;
};

dist_qoi_statistics calculate_statistics(
    const std::vector<double>& dists,
    const std::vector<double>& orig_data,
    const std::vector<double>& dec_data) {
  dist_qoi_statistics stats{};
  
  if (!dists.empty()) {
    double sum = 0.0;
    for (size_t i = 0; i < dists.size(); ++i) {
      sum += dists[i];
    }
    stats.mean = sum / static_cast<double>(dists.size());
    
    std::vector<double> sorted_dists = dists;
    std::sort(sorted_dists.begin(), sorted_dists.end());
    
    size_t n = sorted_dists.size();
    
    stats.min_val = sorted_dists[0];
    stats.max_val = sorted_dists[n - 1];
    
    if (n % 2 == 0) {
      stats.median = (sorted_dists[n/2 - 1] + sorted_dists[n/2]) / 2.0;
    } else {
      stats.median = sorted_dists[n/2];
    }
    
    auto percentile = [&sorted_dists, n](double p) -> double {
      if (n == 1) return sorted_dists[0];
      double index = p * (n - 1);
      size_t lower = static_cast<size_t>(index);
      size_t upper = lower + 1;
      if (upper >= n) return sorted_dists[n - 1];
      double fraction = index - lower;
      return sorted_dists[lower] * (1.0 - fraction) + sorted_dists[upper] * fraction;
    };
    
    stats.p90 = percentile(0.90);
    stats.p99 = percentile(0.99);
    stats.p999 = percentile(0.999);
  }
  
  if (!orig_data.empty() && !dec_data.empty()) {
    std::vector<double> u = orig_data;
    std::vector<double> v = dec_data;
    std::sort(u.begin(), u.end());
    std::sort(v.begin(), v.end());

    const size_t u_size = u.size();
    const size_t v_size = v.size();
    size_t i = 0;
    size_t j = 0;
    double cdf_u = 0.0;
    double cdf_v = 0.0;
    double x_prev = 0.0;
    bool has_prev = false;
    double w = 0.0;

    // Merge-scan both sorted arrays and integrate |F_u(x) - F_v(x)| dx.
    while (i < u_size || j < v_size) {
      double x_curr;
      if (j >= v_size || (i < u_size && u[i] <= v[j])) {
        x_curr = u[i];
      } else {
        x_curr = v[j];
      }

      if (has_prev) {
        w += std::abs(cdf_u - cdf_v) * (x_curr - x_prev);
      } else {
        has_prev = true;
      }

      while (i < u_size && u[i] == x_curr) ++i;
      while (j < v_size && v[j] == x_curr) ++j;

      cdf_u = static_cast<double>(i) / static_cast<double>(u_size);
      cdf_v = static_cast<double>(j) / static_cast<double>(v_size);
      x_prev = x_curr;
    }

    stats.wasserstein_distance = w;

  }
  
  return stats;
}



class dist_qoi_plugin : public libpressio_metrics_plugin {
  public:
  int set_options(pressio_options const& options) override {
    get_meta(options, "dist_qoi:metric", metrics_plugins(), child_id, child);
    get(options, "dist_qoi:orig_data_key", &orig_data_key);
    get(options, "dist_qoi:dec_data_key", &dec_data_key);
    return 0;
  }
  pressio_options get_options() const override {
    pressio_options opts;
    set_meta(opts, "dist_qoi:metric", child_id, child);
    set(opts, "dist_qoi:orig_data_key", orig_data_key);
    set(opts, "dist_qoi:dec_data_key", dec_data_key);
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
    set_meta_configuration(opts, "dist_qoi:metric", metrics_plugins(), child);
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(opts, "predictors:requires_decompress", true);
    set(opts, "predictors:invalidate", std::vector<std::string>{"predictors:error_dependent"});
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set_meta_docs(opt, "dist_qoi:metric", "child metrics plugin supplying the configured QoI data arrays", child);
    set(opt, "pressio:description", "computes distance statistics and Wasserstein distance from configured child metrics results");
    set(opt, "dist_qoi:data", "elementwise absolute differences between original and decompressed QoI data");
    set(opt, "dist_qoi:mean", "arithmetic mean of the elementwise absolute QoI errors");
    set(opt, "dist_qoi:min", "minimum elementwise absolute QoI error");
    set(opt, "dist_qoi:max", "maximum elementwise absolute QoI error");
    set(opt, "dist_qoi:median", "median elementwise absolute QoI error; mean of the two middle values for an even sample count");
    set(opt, "dist_qoi:p90", "90th percentile of elementwise absolute QoI errors using linear interpolation at 0.90 * (n - 1)");
    set(opt, "dist_qoi:p99", "99th percentile of elementwise absolute QoI errors using linear interpolation at 0.99 * (n - 1)");
    set(opt, "dist_qoi:p999", "99.9th percentile of elementwise absolute QoI errors using linear interpolation at 0.999 * (n - 1)");
    set(opt, "dist_qoi:wasserstein_distance", "1-Wasserstein distance between the equally weighted empirical orig_data and dec_data samples");
    set(opt, "dist_qoi:orig_data_key", "child result key for original QoI data; empty disables lookup and paired metrics");
    set(opt, "dist_qoi:dec_data_key", "child result key for decompressed QoI data; empty disables lookup and paired metrics");
    set(opt, "dist_qoi:error_code", "0 on success, 1 for non-finite data or absolute error, 2 for a missing, unset, or incompatible configured child result, 3 for unequal array sizes; no QoI results are computed on error");
    set(opt, "dist_qoi:error_msg", "diagnostic identifying the configured child result key and, for non-finite input, its zero-based element index; empty on success");
    return opt;

  }

  pressio_options get_metrics_results(pressio_options const & parent) override {
    pressio_options opt = child->get_metrics_results(parent);

    std::vector<double> values;
    std::vector<double> orig_data;
    std::vector<double> dec_data;

    // Keep input lookup independent of the external metrics namespace.
    // Only explicitly configured inputs participate in local calculations.
    auto lookup_data = [&opt, this](std::string const& key, std::vector<double>& data) {
      if (key.empty()) return true;
      pressio_data input;
      auto status = opt.get(child->get_name(), key, &input);
      if (status != pressio_options_key_set) {
        set(opt, "dist_qoi:error_code", int32_t{2});
        set(opt, "dist_qoi:error_msg", "configured child result key '" + key +
            "' is missing, unset, or is not pressio_data");
        return false;
      }
      if (input.dtype() != pressio_double_dtype ||
          (input.num_elements() > 0 && input.data() == nullptr)) {
        set(opt, "dist_qoi:error_code", int32_t{2});
        set(opt, "dist_qoi:error_msg", "configured child result key '" + key +
            "' must contain double-precision data with a valid buffer");
        return false;
      }
      if (input.num_elements() > 0) {
        auto ptr = static_cast<const double*>(input.data());
        data.assign(ptr, ptr + input.num_elements());
      }
      return true;
    };
    if (!lookup_data(orig_data_key, orig_data) ||
        !lookup_data(dec_data_key, dec_data)) {
      return opt;
    }

    set(opt, "dist_qoi:error_code", int32_t{0});
    set(opt, "dist_qoi:error_msg", std::string{});

    // Reject the entire QoI calculation before sorting or invoking external code.
    // In particular, NaN would prevent the Wasserstein merge scan from advancing.
    auto validate_finite = [&opt, this](std::string const& key, const std::vector<double>& data) {
      auto invalid = std::find_if(data.begin(), data.end(), [](double value) {
        return !std::isfinite(value);
      });
      if (invalid == data.end()) return true;
      set(opt, "dist_qoi:error_code", int32_t{1});
      set(opt, "dist_qoi:error_msg", std::string(key) + " contains NaN or infinity at index " +
          std::to_string(static_cast<size_t>(invalid - data.begin())));
      return false;
    };
    if (!validate_finite(orig_data_key, orig_data) ||
        !validate_finite(dec_data_key, dec_data)) {
      return opt;
    }

    if (orig_data_key.empty() || dec_data_key.empty()) return opt;
    if (orig_data.size() != dec_data.size()) {
      set(opt, "dist_qoi:error_code", int32_t{3});
      set(opt, "dist_qoi:error_msg", "original and decompressed QoI arrays must have the same number of elements");
      return opt;
    }
    values.reserve(orig_data.size());
    for (size_t i = 0; i < orig_data.size(); ++i) {
      double error = std::abs(orig_data[i] - dec_data[i]);
      if (!std::isfinite(error)) {
        set(opt, "dist_qoi:error_code", int32_t{1});
        set(opt, "dist_qoi:error_msg", "absolute QoI error is non-finite at index " + std::to_string(i));
        return opt;
      }
      values.push_back(error);
    }

    dist_qoi_statistics stats = calculate_statistics(values, orig_data, dec_data);
    if (!values.empty()) {
      set(opt, "dist_qoi:mean", stats.mean);
      set(opt, "dist_qoi:min", stats.min_val);
      set(opt, "dist_qoi:max", stats.max_val);
      set(opt, "dist_qoi:median", stats.median);
      set(opt, "dist_qoi:p90", stats.p90);
      set(opt, "dist_qoi:p99", stats.p99);
      set(opt, "dist_qoi:p999", stats.p999);
      set(opt, "dist_qoi:data", pressio_data(values.begin(), values.end()));
    }
    if (!orig_data.empty() && !dec_data.empty()) {
      set(opt, "dist_qoi:wasserstein_distance", stats.wasserstein_distance);
    }

    return opt;
  }


  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<dist_qoi_plugin>(*this);
  }

  const char* prefix() const override {
    return "dist_qoi";
  }



  private:

  pressio_metrics child = metrics_plugins().build("noop");
  std::string child_id = "noop";
  std::string orig_data_key;
  std::string dec_data_key;
};

pressio_register registration(metrics_plugins(), "dist_qoi", [](){ return compat::make_unique<dist_qoi_plugin>(); });

}}}
