#include <cmath>
#include <algorithm>
#include "pressio_data.h"
#include "pressio_options.h"
#include "pressio_compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"

namespace {
  

  struct compute_metrics{
    template <class RandomIt1, class RandomIt2>
    double operator()(RandomIt1 input_begin, RandomIt1 input_end,
                      RandomIt2 decomp_begin, RandomIt2 decomp_end)
    {
      size_t total_elements = std::min(std::distance(input_begin, input_end), std::distance(decomp_begin, decomp_end));
      size_t out_of_bounds = 0;
      #pragma omp parallel for reduction(+:out_of_bounds)
      for (size_t i = 0; i < total_elements; ++i) {
        double error;
        auto original = input_begin[i];
        auto reconstructed = decomp_begin[i];
        if(original == 0) {
          error = std::fabs(original - reconstructed);
        } else {
          error = (original - reconstructed) / static_cast<double>(original);
        }
        if(error > threshold) {
          out_of_bounds++;
        }
      }
      return out_of_bounds / static_cast<double>(total_elements) * 100.0;
    }

    double threshold;
  };
}

class spatial_error_plugin : public libpressio_metrics_plugin
{

public:
  int begin_compress_impl(const struct pressio_data* input,
                      struct pressio_data const*) override
  {
    input_data = pressio_data::clone(*input);
    return 0;
  }
  int end_decompress_impl(struct pressio_data const*,
                      struct pressio_data const* output, int) override
  {
    spatial_error =
      pressio_data_for_each<double>(input_data, *output, compute_metrics{threshold});
    return 0;
  }

  struct pressio_options get_configuration() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_multiple));
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opts;
    set(opts, "pressio:description", "Computes the Spatial Error Percentage -- the percentage of elements that exceed some threshold");
    set(opts, "spatial_error:threshold", "threshold for triggering spatial_error");
    set(opts, "spatial_error:percent", "percentage of elements that exceed the threshold");
    return opts;
  }

  struct pressio_options get_options() const override {
    pressio_options opts;
    set(opts, "spatial_error:threshold", threshold);
    return opts;
  }

  int set_options(pressio_options const& options) override {
    get(options, "spatial_error:threshold", &threshold);
    return 0;
  }

  pressio_options get_metrics_results(pressio_options const &) const override
  {
    pressio_options opt;
    if (spatial_error) {
      set(opt, "spatial_error:percent", *spatial_error);
    } else {
      set_type(opt, "spatial_error:percent", pressio_option_double_type);
    }
    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<spatial_error_plugin>(*this);
  }
  const char* prefix() const override {
    return "spatial_error";
  }

private:
  pressio_data input_data = pressio_data::empty(pressio_byte_dtype, {});
  compat::optional<double> spatial_error;
  double threshold = .01;
};

static pressio_register metrics_spatial_error_plugin(metrics_plugins(), "spatial_error", []() {
  return compat::make_unique<spatial_error_plugin>();
});
