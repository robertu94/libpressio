#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/data.h"
#include "std_compat/memory.h"

#include <cstring>
#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>






namespace libpressio { namespace metrics {

namespace qoi_ns {

struct qoi_statistics {
  double mean;
  double min_val;
  double max_val;
  double median;
  double p90;
  double p99;
  double p999;
  double wasserstein_distance;
};

qoi_statistics calculate_statistics(
    const std::vector<double>& dists,
    const std::vector<double>& mass_orig,
    const std::vector<double>& mass_dec) {
  qoi_statistics stats = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  
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
  
  if (!mass_orig.empty() && !mass_dec.empty()) {
    std::vector<double> u = mass_orig;
    std::vector<double> v = mass_dec;
    std::sort(u.begin(), u.end());
    std::sort(v.begin(), v.end());
    
    std::vector<double> all;
    all.insert(all.end(), u.begin(), u.end());
    all.insert(all.end(), v.begin(), v.end());
    std::sort(all.begin(), all.end());
    
    double w = 0.0;
    size_t u_size = u.size();
    size_t v_size = v.size();
    
    for (size_t i = 0; i < all.size() - 1; ++i) {
      double x = all[i];
      double dx = all[i + 1] - all[i];
      
      size_t count_u = 0;
      for (size_t j = 0; j < u_size; ++j) {
        if (u[j] <= x) {
          count_u++;
        } else {
          break;
        }
      }
      
      size_t count_v = 0;
      for (size_t j = 0; j < v_size; ++j) {
        if (v[j] <= x) {
          count_v++;
        } else {
          break;
        }
      }
      
      double U = static_cast<double>(count_u) / static_cast<double>(u_size);
      double V = static_cast<double>(count_v) / static_cast<double>(v_size);
      
      w += std::abs(U - V) * dx;
    }
    
    stats.wasserstein_distance = w;
  }
  
  return stats;
}

class qoi_plugin : public libpressio_metrics_plugin {
  public:
  int set_options(pressio_options const& options) override {
    get_meta(options, "qoi:metric", metrics_plugins(), child_id, child);
    return 0;
  }
  pressio_options get_options() const override {
    pressio_options opts;
    set_meta(opts, "qoi:metric", child_id, child);
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

    values.clear();
    mass_orig.clear();
    mass_dec.clear();

    bool has_dists = false, has_mass_orig = false, has_mass_dec = false;

    for (auto const& item : opt) {
      const std::string& key = item.first;
      const pressio_option& option = item.second;
      
      if (key.find(qoi) == 0) {
        if (key == "external:results:dists") {
          if (option.holds_alternative<pressio_data>() && option.has_value()) {
            pressio_data temp_data = option.get_value<pressio_data>();
            const double* ptr = static_cast<const double*>(temp_data.data());
            size_t n = temp_data.num_elements();
            if (ptr != nullptr && n > 0 && temp_data.dtype() == pressio_double_dtype) {
              values.assign(ptr, ptr + n);
              has_dists = true;
            }
          }
        } 
        else if (key == "external:results:mass_orig") {
          if (option.holds_alternative<pressio_data>() && option.has_value()) {
            pressio_data temp_data = option.get_value<pressio_data>();
            const double* ptr = static_cast<const double*>(temp_data.data());
            size_t n = temp_data.num_elements();
            if (ptr != nullptr && n > 0 && temp_data.dtype() == pressio_double_dtype) {
              mass_orig.assign(ptr, ptr + n);
              has_mass_orig = true;
            }
          }
        } else if (key == "external:results:mass_dec") {
          if (option.holds_alternative<pressio_data>() && option.has_value()) {
            pressio_data temp_data = option.get_value<pressio_data>();
            const double* ptr = static_cast<const double*>(temp_data.data());
            size_t n = temp_data.num_elements();
            if (ptr != nullptr && n > 0 && temp_data.dtype() == pressio_double_dtype) {
              mass_dec.assign(ptr, ptr + n);
              has_mass_dec = true;
            }
          }
        }
      }
    }
    std::cout << "[QOI] Total values in vector: " << values.size() << std::endl;

    if (!values.empty()) {
      qoi_statistics stats = calculate_statistics(values, mass_orig, mass_dec);
      
      std::cout << "[QOI] Statistics:" << std::endl;
      std::cout << "[QOI]   mean:   " << stats.mean << std::endl;
      std::cout << "[QOI]   min:    " << stats.min_val << std::endl;
      std::cout << "[QOI]   max:    " << stats.max_val << std::endl;
      std::cout << "[QOI]   median: " << stats.median << std::endl;
      std::cout << "[QOI]   p90:    " << stats.p90 << std::endl;
      std::cout << "[QOI]   p99:    " << stats.p99 << std::endl;
      std::cout << "[QOI]   p999:   " << stats.p999 << std::endl;
      
      if (has_mass_orig && has_mass_dec) {
        std::cout << "[QOI]   wasserstein_distance: " << stats.wasserstein_distance << std::endl;
      }
      
      qoi_data = pressio_data::copy(pressio_double_dtype, values.data(), {values.size()});
      set(opt, "qoi:data", qoi_data);
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

  std::string qoi = "external:results";
  pressio_metrics child = metrics_plugins().build("noop");
  std::string child_id = "noop";
  pressio_data qoi_data = pressio_data::empty(pressio_byte_dtype, {}); 
  std::vector<double> values;
  std::vector<double> mass_orig;
  std::vector<double> mass_dec;
};

pressio_register registration(metrics_plugins(), "qoi", [](){   std::cout << "Registering qoi plugin\n"; return compat::make_unique<qoi_plugin>(); });

}}}
