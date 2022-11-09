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
namespace autocorr {
  struct metrics {
    pressio_data autocorr;
  };
  struct compute_metrics{
    template <class ForwardIt1, class ForwardIt2>
      metrics operator()(ForwardIt1 input_begin, ForwardIt1 input_end, ForwardIt2 input2_begin, ForwardIt2 input2_end) const
      {
        metrics m;
        const size_t num_elements = std::min(input_end - input_begin, input2_end- input2_begin);
        if(num_elements == 0) {
          return m;
        }

        std::vector<double> errors(num_elements);

        m.autocorr = pressio_data::owning(pressio_double_dtype, {static_cast<size_t>(autocorr_lags + 1)});
        double average_error=0;
        for (size_t i = 0; i < num_elements; ++i) {
          errors[i] = input_begin[i] - input2_begin[i];
          average_error += errors[i];
        }
        average_error /= static_cast<double>(num_elements);
        auto autocorr = static_cast<double*>(m.autocorr.data());


        if (num_elements > 4096)
        {
          double cov = 0;
          for (size_t i = 0; i < num_elements; i++) {
            cov += (errors[i] - average_error)*(errors[i] - average_error);
          }

          cov = cov/num_elements;

          if (cov == 0)
          {
            for (size_t delta = 1; delta <= autocorr_lags; delta++)
              autocorr[delta] = 0;
          }
          else
          {
            for(size_t delta = 1; delta <= autocorr_lags; delta++)
            {
              double sum = 0;

              for (size_t i = 0; i < num_elements-delta; i++)
                sum += (errors[i]-average_error)*(errors[i+delta]-average_error);

              autocorr[delta] = sum/(num_elements-delta)/cov;
            }
          }
        }
        else
        {
          for (size_t delta = 1; delta <= autocorr_lags; delta++)
          {
            double avg_0 = 0;
            double avg_1 = 0;

            for (size_t i = 0; i < num_elements-delta; i++)
            {
              avg_0 += errors[i];
              avg_1 += errors[i+delta];
            }

            avg_0 = avg_0 / (num_elements-delta);
            avg_1 = avg_1 / (num_elements-delta);

            double cov_0 = 0;
            double cov_1 = 0;

            for (size_t i = 0; i < num_elements-delta; i++)
            {
              cov_0 += (errors[i]-avg_0)*(errors[i]-avg_0);
              cov_1 += (errors[i+delta]-avg_1)*(errors[i+delta]-avg_1);
            }

            cov_0 = cov_0/(num_elements-delta);
            cov_1 = cov_1/(num_elements-delta);

            cov_0 = sqrt(cov_0);
            cov_1 = sqrt(cov_1);

            if (cov_0*cov_1 == 0)
            {
              for (delta = 1; delta <= autocorr_lags; delta++)
                autocorr[delta] = 1;
            }
            else
            {
              double sum = 0;

              for (size_t i = 0; i < num_elements-delta; i++)
                sum += (errors[i]-avg_0)*(errors[i+delta]-avg_1);

              autocorr[delta] = sum/(num_elements-delta)/(cov_0*cov_1);
            }
          }
        }

        autocorr[0] = 1;


        return m;
      }

    uint64_t autocorr_lags;
  };

class autocorr_plugin : public libpressio_metrics_plugin {

  public:
    int begin_compress_impl(const struct pressio_data * input, struct pressio_data const * ) override {
      input_data = pressio_data::clone(*input);
      return 0;
    }
    int end_decompress_impl(struct pressio_data const*, struct pressio_data const* output, int ) override {
      err_metrics = pressio_data_for_each<autocorr::metrics>(input_data, *output, autocorr::compute_metrics{autocorr_lags});
      return 0;
    }

    int set_options(pressio_options const& opts) override {
      get(opts, "autocorr:autocorr_lags", &autocorr_lags);
      return 0;
    }
    pressio_options get_options() const override {
      pressio_options opts;
      set(opts, "autocorr:autocorr_lags", autocorr_lags);
      return opts;
    }

    struct pressio_options get_configuration() const override {
      pressio_options opts;
      set(opts, "pressio:stability", "stable");
      set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
      return opts;
    }

    struct pressio_options get_documentation_impl() const override {
      pressio_options opts;
      set(opts, "autocorr:autocorr_lags", "how many autocorrelation lags to compute");
      set(opts, "autocorr:autocorr", "the 1d autocorrelation");
      set(opts, "pressio:description", "computes the 1d autocorrelation");
      return opts;
    }
    pressio_options get_metrics_results(pressio_options const&)  override {
      pressio_options opt;
      if(err_metrics) {
        set(opt, "autocorr:autocorr", err_metrics->autocorr);
      } else {
        set_type(opt, "autocorr:autocorr", pressio_option_data_type);
      }
      return opt;
    }
    std::unique_ptr<libpressio_metrics_plugin> clone() override {
      return compat::make_unique<autocorr_plugin>(*this);
    }

  const char* prefix() const override {
    return "autocorr";
  }


  private:
  uint64_t autocorr_lags = 100;
  pressio_data input_data = pressio_data::empty(pressio_byte_dtype, {});
  compat::optional<autocorr::metrics> err_metrics;
};

static pressio_register metrics_autocorr_plugin(metrics_plugins(), "autocorr", [](){ return compat::make_unique<autocorr_plugin>(); });
} }
