#include <cmath>
#include <algorithm>
#include "pressio_data.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/compat/std_compat.h"

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
      return out_of_bounds / static_cast<double>(total_elements);
    }

    double threshold;
  };
}

class spatial_error_plugin : public libpressio_metrics_plugin
{

public:
  void begin_compress(const struct pressio_data* input,
                      struct pressio_data const*) override
  {
    input_data = pressio_data::clone(*input);
  }
  void end_decompress(struct pressio_data const*,
                      struct pressio_data const* output, int) override
  {
    spatial_error =
      pressio_data_for_each<double>(input_data, *output, compute_metrics{threshold});
  }

  struct pressio_options get_metrics_options() const override {
    pressio_options opts;
    opts.set("spatial_error:threshold", threshold);
    return opts;
  }

  int set_metrics_options(pressio_options const& options) override {
    options.get("spatial_error:threshold", &threshold);
    return 0;
  }

  struct pressio_options get_metrics_results() const override
  {
    pressio_options opt;
    if (spatial_error) {
      opt.set("spatial_error:percent", *spatial_error);
    } else {
      opt.set_type("spatial_error:percent", pressio_option_double_type);
    }
    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<spatial_error_plugin>(*this);
  }

private:
  pressio_data input_data = pressio_data::empty(pressio_byte_dtype, {});
  compat::optional<double> spatial_error;
  double threshold = 1e-4;
};

static pressio_register X(metrics_plugins(), "spatial_error", []() {
  return compat::make_unique<spatial_error_plugin>();
});
