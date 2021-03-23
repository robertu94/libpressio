#include "pressio_data.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"

namespace diff_pdf {
  static const uint64_t zero = 0;
  struct metrics {
    pressio_data histogram = pressio_data::copy(pressio_uint64_dtype, &zero, {1});
    double interval = 0;
    double min_diff = 0;
    double max_diff = 0;
  };
  struct compute_metrics{
    template <class ForwardIt1, class ForwardIt2>
    metrics operator()(ForwardIt1 input_begin, ForwardIt1 input_end, ForwardIt2 input2_begin, ForwardIt2 input2_end) const
    {
      metrics m;
      size_t num_elements = std::min(input_end-input_begin, input2_end-input2_begin);
      if (num_elements == 0) {
        return m;
      }

      m.max_diff = input_begin[0] - input2_begin[0];
      m.min_diff = m.max_diff;
      for (size_t i = 1; i < num_elements; ++i) {
        double diff = input_begin[i] - input2_begin[i];
        m.max_diff = std::max(m.max_diff, diff);
        m.min_diff = std::min(m.min_diff, diff);
      }
      m.interval = (m.max_diff - m.min_diff)/pdf_intervals;
      if (m.interval == 0) {
        return m;
      }


      m.histogram = pressio_data::owning(pressio_uint64_dtype, {static_cast<size_t>(pdf_intervals)});
      const auto dist_ptr = static_cast<uint64_t*>(m.histogram.data());
      memset(dist_ptr, 0, m.histogram.size_in_bytes());

      for (size_t i = 0; i < num_elements; ++i) {
        auto diff = input_begin[i] - input2_begin[i];
        auto idx = std::min(static_cast<size_t>((diff-m.min_diff)/m.interval), static_cast<size_t>(pdf_intervals - 1));
        dist_ptr[idx]++;
      }

      return m;

    }

    uint64_t pdf_intervals;
  };
}

class diff_pdf_plugin : public libpressio_metrics_plugin {

  public:
    void begin_compress(const struct pressio_data * input, struct pressio_data const * ) override {
      input_data = pressio_data::clone(*input);
    }
    void end_decompress(struct pressio_data const*, struct pressio_data const* output, int ) override {
      err_metrics = pressio_data_for_each<diff_pdf::metrics>(input_data, *output, diff_pdf::compute_metrics{pdf_intervals});      
    }

    int set_options(pressio_options const& opts) override {
      get(opts, "diff_pdf:intervals", &pdf_intervals);
      return 0;
    }
    pressio_options get_options() const override {
      pressio_options opts;
      set(opts, "diff_pdf:intervals", pdf_intervals);
      return opts;
    }

    struct pressio_options get_metrics_results() const override {
      pressio_options opt;
      if(err_metrics) {
        set(opt, "diff_pdf:histogram", err_metrics->histogram);
        set(opt, "diff_pdf:interval", err_metrics->interval);
        set(opt, "diff_pdf:min_diff", err_metrics->min_diff);
        set(opt, "diff_pdf:max_diff", err_metrics->max_diff);
      } else {
        set_type(opt, "diff_pdf:pdf", pressio_option_data_type);
      }
      return opt;
    }
    std::unique_ptr<libpressio_metrics_plugin> clone() override {
      return compat::make_unique<diff_pdf_plugin>(*this);
    }

  const char* prefix() const override {
    return "diff_pdf";
  }


  private:
  uint64_t pdf_intervals = 2000;
  pressio_data input_data = pressio_data::empty(pressio_byte_dtype, {});
  compat::optional<diff_pdf::metrics> err_metrics;
};

static pressio_register metrics_diff_pdf_plugin(metrics_plugins(), "diff_pdf", [](){ return compat::make_unique<diff_pdf_plugin>(); });
