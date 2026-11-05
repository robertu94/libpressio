#include "pressio_options.h"
#include "pressio_version.h"
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
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cmath>
#include <limits>
#include <unistd.h>

#if LIBPRESSIO_HAS_JSON
#include <nlohmann/json.hpp>
#endif






namespace libpressio { namespace metrics {

namespace qoi_ns {

double compute_dssim_from_vectors(const std::vector<double>& orig, const std::vector<double>& dec, size_t height, size_t width);
double compute_fidelity_from_vectors(const std::vector<double>& orig, const std::vector<double>& dec);

struct qoi_statistics {
  double mean;
  double min_val;
  double max_val;
  double median;
  double p90;
  double p99;
  double p999;
  double wasserstein_distance;
  double dssim;
  double fidelity;
};

qoi_statistics calculate_statistics(
    const std::vector<double>& dists,
    const std::vector<double>& mass_orig,
    const std::vector<double>& mass_dec) {
  qoi_statistics stats = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
  
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

double compute_dssim_from_vectors(const std::vector<double>& orig, const std::vector<double>& dec, size_t height, size_t width) {
  if (orig.empty() || orig.size() != dec.size()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (height == 0 || width == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (orig.size() != height * width) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  // Call Python implementation in DSSIM/compute_dssim.py with orig=mass_orig, dec=mass_dec.
  char orig_tpl[] = "/tmp/qoi_mass_orig_XXXXXX";
  char dec_tpl[]  = "/tmp/qoi_mass_dec_XXXXXX";
  int orig_fd = mkstemp(orig_tpl);
  int dec_fd = mkstemp(dec_tpl);
  if (orig_fd == -1 || dec_fd == -1) {
    if (orig_fd != -1) { close(orig_fd); std::remove(orig_tpl); }
    if (dec_fd != -1) { close(dec_fd); std::remove(dec_tpl); }
    return std::numeric_limits<double>::quiet_NaN();
  }
  close(orig_fd);
  close(dec_fd);

  {
    std::ofstream fo(orig_tpl, std::ios::binary);
    std::ofstream fd(dec_tpl, std::ios::binary);
    if (!fo || !fd) {
      std::remove(orig_tpl);
      std::remove(dec_tpl);
      return std::numeric_limits<double>::quiet_NaN();
    }
    fo.write(reinterpret_cast<const char*>(orig.data()), static_cast<std::streamsize>(orig.size() * sizeof(double)));
    fd.write(reinterpret_cast<const char*>(dec.data()), static_cast<std::streamsize>(dec.size() * sizeof(double)));
  }

  const std::string py = "/anvil/projects/x-cis240669/DSSIM/dssim-env/bin/python";
  const std::string script = "/anvil/projects/x-cis240669/DSSIM/compute_dssim.py";
  std::string cmd = "env -u PYTHONPATH " + py + " " + script +
                    " --orig " + std::string(orig_tpl) +
                    " --dec " + std::string(dec_tpl) +
                    " --width " + std::to_string(static_cast<unsigned long long>(width)) +
                    " --height " + std::to_string(static_cast<unsigned long long>(height));

  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    std::remove(orig_tpl);
    std::remove(dec_tpl);
    return std::numeric_limits<double>::quiet_NaN();
  }

  std::string out;
  char buffer[256];
  while (fgets(buffer, sizeof(buffer), pipe)) {
    out += buffer;
  }
  int rc = pclose(pipe);

  std::remove(orig_tpl);
  std::remove(dec_tpl);

  if (rc != 0 || out.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  try {
    return std::stod(out);
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

double compute_fidelity_from_vectors(const std::vector<double>& orig, const std::vector<double>& dec) {
  if (orig.empty() || orig.size() != dec.size()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (orig.size() % 2 != 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  // orig/dec are interleaved [r0, i0, r1, i1, ...] representing complex vectors.
  // fidelity = |vdot(ref, rec)| / (||ref|| * ||rec||)
  // where vdot(a, b) = sum(conj(a_k) * b_k)
  const size_t n = orig.size() / 2;
  double norm_ref_sq = 0.0;
  double norm_rec_sq = 0.0;
  double dot_real = 0.0;
  double dot_imag = 0.0;

  for (size_t k = 0; k < n; ++k) {
    const double ar = orig[2 * k];
    const double ai = orig[2 * k + 1];
    const double br = dec[2 * k];
    const double bi = dec[2 * k + 1];
    // conj(a) * b = (ar*br + ai*bi) + j*(ar*bi - ai*br)
    dot_real += ar * br + ai * bi;
    dot_imag += ar * bi - ai * br;
    norm_ref_sq += ar * ar + ai * ai;
    norm_rec_sq += br * br + bi * bi;
  }

  const double norm_ref = std::sqrt(norm_ref_sq);
  const double norm_rec = std::sqrt(norm_rec_sq);
  if (norm_ref == 0.0 || norm_rec == 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  const double abs_dot = std::sqrt(dot_real * dot_real + dot_imag * dot_imag);
  double fidelity = abs_dot / (norm_ref * norm_rec);
  if (fidelity > 1.0) fidelity = 1.0;
  return fidelity;
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
    mirrored_qoi_metrics.clear();

    bool has_dists = false, has_mass_orig = false, has_mass_dec = false;
    size_t mass_height = 0, mass_width = 0;
    std::string metrics_file_path;

    for (auto const& item : opt) {
      const std::string& key = item.first;
      const pressio_option& option = item.second;
      
      if (key.find(qoi) == 0) {
        if (key.size() > qoi.size() && key[qoi.size()] == ':') {
          auto metric_name = key.substr(qoi.size() + 1);
          if (!metric_name.empty()) {
            mirrored_qoi_metrics.emplace_back("qoi:" + metric_name, option);
          }
        }
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
              auto dims = temp_data.dimensions();
              if (dims.size() >= 2) {
                mass_height = dims[0];
                mass_width = dims[1];
              } else if (dims.size() == 1) {
                mass_height = 1;
                mass_width = dims[0];
              }
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
              if (mass_height == 0 || mass_width == 0) {
                auto dims = temp_data.dimensions();
                if (dims.size() >= 2) {
                  mass_height = dims[0];
                  mass_width = dims[1];
                } else if (dims.size() == 1) {
                  mass_height = 1;
                  mass_width = dims[0];
                }
              }
            }
          }
        } else if (key == "external:results:metrics_file") {
          if (option.holds_alternative<std::string>() && option.has_value()) {
            metrics_file_path = option.get_value<std::string>();
          }
        }
      }
    }

#if LIBPRESSIO_HAS_JSON
    if (!metrics_file_path.empty() && (!has_dists || !has_mass_orig || !has_mass_dec)) {
      std::ifstream f(metrics_file_path);
      if (f) {
        try {
          nlohmann::json j_file = nlohmann::json::parse(f);
          if (j_file.contains("dists") && j_file["dists"].is_array()) {
            values = j_file["dists"].get<std::vector<double>>();
            has_dists = true;
          }
          if (j_file.contains("mass_orig") && j_file["mass_orig"].is_array()) {
            mass_orig = j_file["mass_orig"].get<std::vector<double>>();
            has_mass_orig = true;
          }
          if (j_file.contains("mass_dec") && j_file["mass_dec"].is_array()) {
            mass_dec = j_file["mass_dec"].get<std::vector<double>>();
            has_mass_dec = true;
          }
        } catch (...) {}
      }
    }
#endif
    std::cout << "[QOI] Total values in vector: " << values.size() << std::endl;

    if (!values.empty()) {
      qoi_statistics stats = calculate_statistics(values, mass_orig, mass_dec);
      if (has_mass_orig && has_mass_dec && mass_orig.size() == mass_dec.size()) {
        if (mass_height > 0 && mass_width > 0) {
          stats.dssim = compute_dssim_from_vectors(mass_orig, mass_dec, mass_height, mass_width);
        }
        stats.fidelity = compute_fidelity_from_vectors(mass_orig, mass_dec);
      }
      
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
        if (std::isfinite(stats.dssim)) {
          std::cout << "[QOI]   dssim:  " << stats.dssim << std::endl;
          set(opt, "qoi:dssim", stats.dssim);
        }
        if (std::isfinite(stats.fidelity)) {
          std::cout << "[QOI]   fidelity: " << std::setprecision(15) << stats.fidelity << std::endl;
          set(opt, "qoi:fidelity", stats.fidelity);
        }
      }
      
      qoi_data = pressio_data::copy(pressio_double_dtype, values.data(), {values.size()});
      set(opt, "qoi:data", qoi_data);
    }

    // Mirror all external qoi metrics into the qoi namespace so callers can
    // consume qoi:* directly (e.g., qoi:dssim) without parsing external keys.
    for (auto const& metric : mirrored_qoi_metrics) {
      set(opt, metric.first, metric.second);
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
  std::vector<std::pair<std::string, pressio_option>> mirrored_qoi_metrics;
};

pressio_register registration(metrics_plugins(), "qoi", [](){   std::cout << "Registering qoi plugin\n"; return compat::make_unique<qoi_plugin>(); });

}}}
#if 0
#include "pressio_options.h"
#include "pressio_version.h"
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
#include <fstream>
#include <cstdint>
#include <cmath>
#include <limits>

#if LIBPRESSIO_HAS_JSON
#include <nlohmann/json.hpp>
#endif






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



template <typename T>
void append_casted(const pressio_data& data, std::vector<double>& out) {
  const T* ptr = static_cast<const T*>(data.data());
  if (ptr == nullptr) return;
  size_t n = data.num_elements();
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    out.push_back(static_cast<double>(ptr[i]));
  }
}

bool to_double_vector(const pressio_data& data, std::vector<double>& out) {
  out.clear();
  if (data.num_elements() == 0 || data.data() == nullptr) return false;

  switch (data.dtype()) {
    case pressio_float_dtype:  append_casted<float>(data, out); break;
    case pressio_double_dtype: append_casted<double>(data, out); break;
    case pressio_bool_dtype:   append_casted<bool>(data, out); break;
    case pressio_int8_dtype:   append_casted<int8_t>(data, out); break;
    case pressio_int16_dtype:  append_casted<int16_t>(data, out); break;
    case pressio_int32_dtype:  append_casted<int32_t>(data, out); break;
    case pressio_int64_dtype:  append_casted<int64_t>(data, out); break;
    case pressio_uint8_dtype:  append_casted<uint8_t>(data, out); break;
    case pressio_uint16_dtype: append_casted<uint16_t>(data, out); break;
    case pressio_uint32_dtype: append_casted<uint32_t>(data, out); break;
    case pressio_uint64_dtype: append_casted<uint64_t>(data, out); break;
    case pressio_byte_dtype:   append_casted<unsigned char>(data, out); break;
    default: return false;
  }
  return !out.empty();
}

double compute_dssim_from_vectors(const std::vector<double>& orig, const std::vector<double>& dec) {
  if (orig.empty() || orig.size() != dec.size()) return std::numeric_limits<double>::quiet_NaN();

  double smin = std::numeric_limits<double>::infinity();
  double smax = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < orig.size(); ++i) {
    smin = std::min(smin, std::min(orig[i], dec[i]));
    smax = std::max(smax, std::max(orig[i], dec[i]));
  }
  if (!std::isfinite(smin) || !std::isfinite(smax)) return std::numeric_limits<double>::quiet_NaN();

  const double range = smax - smin;
  if (range == 0.0) return 0.0;

  // Port the normalization + 8-bit quantization behavior from pressio_dssim.py
  std::vector<double> x;
  std::vector<double> y;
  x.reserve(orig.size());
  y.reserve(dec.size());
  for (size_t i = 0; i < orig.size(); ++i) {
    double xi = (orig[i] - smin) / range;
    double yi = (dec[i] - smin) / range;
    xi = std::round(xi * 255.0) / 255.0;
    yi = std::round(yi * 255.0) / 255.0;
    x.push_back(xi);
    y.push_back(yi);
  }

  double mu_x = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
  double mu_y = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(y.size());
  double sigma_x2 = 0.0, sigma_y2 = 0.0, sigma_xy = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    const double dx = x[i] - mu_x;
    const double dy = y[i] - mu_y;
    sigma_x2 += dx * dx;
    sigma_y2 += dy * dy;
    sigma_xy += dx * dy;
  }
  sigma_x2 /= static_cast<double>(x.size());
  sigma_y2 /= static_cast<double>(y.size());
  sigma_xy /= static_cast<double>(x.size());

  const double K1 = 1e-4;
  const double K2 = 1e-4;
  const double C1 = K1 * K1;
  const double C2 = K2 * K2;

  const double num = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2);
  const double den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2);
  const double ssim = (den == 0.0) ? 1.0 : (num / den);
  const double clamped_ssim = std::max(-1.0, std::min(1.0, ssim));
  return clamped_ssim;
}

bool compute_dssim_from_data(const pressio_data& orig, const pressio_data& dec, double& out_dssim) {
  if (orig.num_elements() == 0 || dec.num_elements() == 0) return false;
  if (orig.num_elements() != dec.num_elements()) return false;

  std::vector<double> orig_vec;
  std::vector<double> dec_vec;
  if (!to_double_vector(orig, orig_vec)) return false;
  if (!to_double_vector(dec, dec_vec)) return false;

  const double dssim = compute_dssim_from_vectors(orig_vec, dec_vec);
  if (!std::isfinite(dssim)) return false;
  out_dssim = dssim;
  return true;
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
    has_computed_dssim = false;
    dssim_value = 0.0;
    begin_input_data = pressio_data::empty(pressio_byte_dtype, {});
    if (input) {
      begin_input_data = pressio_data::clone(*input);
    }
    return child->begin_compress(input, output);
  }

  int end_compress_impl(struct pressio_data const* input, pressio_data const * output, int rc) override {
    return child->end_compress(input, output, rc);
  }

  int begin_decompress_impl(struct pressio_data const* input, pressio_data const* output) override {
    return child->begin_decompress(input, output);
  }

  int end_decompress_impl(struct pressio_data const* input, pressio_data const* output, int rc) override {
    int child_rc = child->end_decompress(input, output, rc);
    if (output && begin_input_data.num_elements() == output->num_elements()) {
      double dssim = 0.0;
      if (compute_dssim_from_data(begin_input_data, *output, dssim)) {
        has_computed_dssim = true;
        dssim_value = dssim;
      }
    }
    return child_rc;
  }

  int begin_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs) override {
    has_computed_dssim = false;
    dssim_value = 0.0;
    begin_input_data_many.clear();
    begin_input_data_many.resize(inputs.size(), pressio_data::empty(pressio_byte_dtype, {}));
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i]) {
        begin_input_data_many[i] = pressio_data::clone(*inputs[i]);
      }
    }
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
    int child_rc = child->end_decompress_many(inputs, outputs, rc);
    double accum = 0.0;
    size_t count = 0;
    const size_t n = std::min(begin_input_data_many.size(), outputs.size());
    for (size_t i = 0; i < n; ++i) {
      if (!outputs[i]) continue;
      double dssim = 0.0;
      if (compute_dssim_from_data(begin_input_data_many[i], *outputs[i], dssim)) {
        accum += dssim;
        count++;
      }
    }
    if (count > 0) {
      has_computed_dssim = true;
      dssim_value = accum / static_cast<double>(count);
    }
    return child_rc;
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
    mirrored_qoi_metrics.clear();

    bool has_dists = false, has_mass_orig = false, has_mass_dec = false;
    std::string metrics_file_path;

    for (auto const& item : opt) {
      const std::string& key = item.first;
      const pressio_option& option = item.second;
      
      if (key.find(qoi) == 0) {
        if (key.size() > qoi.size() && key[qoi.size()] == ':') {
          auto metric_name = key.substr(qoi.size() + 1);
          if (!metric_name.empty()) {
            mirrored_qoi_metrics.emplace_back("qoi:" + metric_name, option);
          }
        }
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
        } else if (key == "external:results:metrics_file") {
          if (option.holds_alternative<std::string>() && option.has_value()) {
            metrics_file_path = option.get_value<std::string>();
          }
        }
      }
    }

#if LIBPRESSIO_HAS_JSON
    if (!metrics_file_path.empty() && (!has_dists || !has_mass_orig || !has_mass_dec)) {
      std::ifstream f(metrics_file_path);
      if (f) {
        try {
          nlohmann::json j_file = nlohmann::json::parse(f);
          if (j_file.contains("dists") && j_file["dists"].is_array()) {
            values = j_file["dists"].get<std::vector<double>>();
            has_dists = true;
          }
          if (j_file.contains("mass_orig") && j_file["mass_orig"].is_array()) {
            mass_orig = j_file["mass_orig"].get<std::vector<double>>();
            has_mass_orig = true;
          }
          if (j_file.contains("mass_dec") && j_file["mass_dec"].is_array()) {
            mass_dec = j_file["mass_dec"].get<std::vector<double>>();
            has_mass_dec = true;
          }
        } catch (...) {}
      }
    }
#endif
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

    // Mirror all external qoi metrics into the qoi namespace so callers can
    // consume qoi:* directly (e.g., qoi:dssim) without parsing external keys.
    for (auto const& metric : mirrored_qoi_metrics) {
      set(opt, metric.first, metric.second);
    }
    if (has_computed_dssim) {
      set(opt, "qoi:dssim", dssim_value);
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
  std::vector<std::pair<std::string, pressio_option>> mirrored_qoi_metrics;
  pressio_data begin_input_data = pressio_data::empty(pressio_byte_dtype, {});
  std::vector<pressio_data> begin_input_data_many;
  bool has_computed_dssim = false;
  double dssim_value = 0.0;
};

pressio_register registration(metrics_plugins(), "qoi", [](){   std::cout << "Registering qoi plugin\n"; return compat::make_unique<qoi_plugin>(); });

}}}
#endif
