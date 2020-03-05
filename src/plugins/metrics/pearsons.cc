#include <cmath>
#include "pressio_data.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/compat/std_compat.h"

namespace {
  struct pearson_metrics {
    double r;
    double r2;
  };

  struct compute_metrics{
    template <class ForwardIt1, class ForwardIt2>
    pearson_metrics operator()(ForwardIt1 input_begin, ForwardIt1 input_end,
                             ForwardIt2 decomp_begin, ForwardIt2 decomp_end)
    {
      using value_type = typename std::iterator_traits<ForwardIt1>::value_type;
      static_assert(std::is_same<typename std::iterator_traits<ForwardIt1>::value_type, value_type>::value, "the iterators must have the same type");
      double input_sum = 0;
      double decomp_sum = 0;
      size_t n = 0;

      {
        //compute means
        auto input_it = input_begin;
        auto decomp_it = decomp_begin;
        while(input_it != input_end && decomp_it != decomp_end) {
          ++n;
          ++input_it;
          ++decomp_it;
        }
      }

      double input_mean = input_sum / n;
      double decomp_mean = decomp_sum / n;
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
          x_xbar_squared_sum += x_xbar * x_xbar;
          y_ybar_squared_sum += y_ybar * y_ybar;
          x_xbar_y_ybar += x_xbar * y_ybar;
          ++n;
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
}

class pearsons_plugin : public libpressio_metrics_plugin
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
    err_metrics = pressio_data_for_each<pearson_metrics>(input_data, *output,
                                                       compute_metrics{});
  }

  struct pressio_options get_metrics_results() const override
  {
    pressio_options opt;
    if (err_metrics) {
      opt.set("pearson:r", (*err_metrics).r);
      opt.set("pearson:r2", (*err_metrics).r2);
    } else {
      opt.set_type("pearson:r", pressio_option_double_type);
      opt.set_type("pearson:r2", pressio_option_double_type);
    }
    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<pearsons_plugin>(*this);
  }

private:
  pressio_data input_data = pressio_data::empty(pressio_byte_dtype, {});
  compat::optional<pearson_metrics> err_metrics;
};

static pressio_register X(metrics_plugins(), "pearson", []() {
  return compat::make_unique<pearsons_plugin>();
});
