#include <cmath>
#include "pressio_data.h"
#include "pressio_options.h"
#include "pressio_compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"

namespace libpressio {
namespace pearson {
  struct pearson_metrics {
    double r = 0.0;
    double r2 = 0.0;
  };

  struct compute_metrics{
    template <class ForwardIt1, class ForwardIt2>
    pearson_metrics operator()(ForwardIt1 input_begin, ForwardIt1 input_end,
                             ForwardIt2 decomp_begin, ForwardIt2 decomp_end)
    {
      double input_sum = 0;
      double decomp_sum = 0;
      size_t n = 0;

      {
        //compute means
        auto input_it = input_begin;
        auto decomp_it = decomp_begin;
        while(input_it != input_end && decomp_it != decomp_end) {
          input_sum += *input_it;
          decomp_sum += *decomp_it;
          ++n;
          ++input_it;
          ++decomp_it;
        }
      }

      double input_mean = input_sum / static_cast<double>(n);
      double decomp_mean = decomp_sum / static_cast<double>(n);
      double x_xbar_squared_sum = 0;
      double y_ybar_squared_sum = 0;
      double x_xbar_y_ybar = 0;

      {
        //compute sums
        auto input_it = input_begin;
        auto decomp_it = decomp_begin;
        while(input_it != input_end && decomp_it != decomp_end) {
          double x_xbar = *input_it - input_mean;
          double y_ybar = *decomp_it - decomp_mean;
          x_xbar_squared_sum += (x_xbar * x_xbar);
          y_ybar_squared_sum += (y_ybar * y_ybar);
          x_xbar_y_ybar += (x_xbar * y_ybar);
          ++input_it;
          ++decomp_it;
        }
      }

      pearson_metrics m;
      m.r = (x_xbar_y_ybar) / (sqrt(x_xbar_squared_sum)* sqrt(y_ybar_squared_sum));
      m.r2 = m.r * m.r;

      return m;
    }
  };

class pearsons_plugin : public libpressio_metrics_plugin
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
    err_metrics = pressio_data_for_each<pearson::pearson_metrics>(input_data, *output,
                                                       pearson::compute_metrics{});
    return 0;
  }

  struct pressio_options get_configuration() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    return opts;
  }


  pressio_options get_documentation_impl() const override {
    pressio_options opts;
    set(opts, "pressio:description", "computes the Pearson's coefficient of correlation and determination");
    set(opts, "pearson:r", "the Pearson's coefficient of correlation");
    set(opts, "pearson:r2", "the Pearson's coefficient of determination");
    return opts;
  }

  pressio_options get_metrics_results(pressio_options const &)  override
  {
    pressio_options opt;
    if (err_metrics) {
      set(opt, "pearson:r", (*err_metrics).r);
      set(opt, "pearson:r2", (*err_metrics).r2);
    } else {
      set_type(opt, "pearson:r", pressio_option_double_type);
      set_type(opt, "pearson:r2", pressio_option_double_type);
    }
    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<pearsons_plugin>(*this);
  }

  const char* prefix() const override {
    return "pearson";
  }

private:
  pressio_data input_data = pressio_data::empty(pressio_byte_dtype, {});
  compat::optional<pearson::pearson_metrics> err_metrics;
};

static pressio_register metrics_pearson_plugin(metrics_plugins(), "pearson", []() {
  return compat::make_unique<pearsons_plugin>();
});
}
}
