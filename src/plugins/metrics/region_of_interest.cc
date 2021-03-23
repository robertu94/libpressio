#include <cmath>
#include <iterator>
#include <stdexcept>
#include "pressio_data.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"
#include "std_compat/numeric.h"

namespace region_of_interest {
  struct region_of_interest_metrics {
    compat::optional<double> input_avg;
    compat::optional<double> input_sum;
    compat::optional<double> decomp_avg;
    compat::optional<double> decomp_sum;
  };

  struct compute_metrics{
    template <class RandomIt1, class RandomIt2>
    region_of_interest_metrics operator()(RandomIt1 input_begin, RandomIt1 ,
                             RandomIt2 decomp_begin, RandomIt2 )
    {
      region_of_interest_metrics m;
      if (start.empty()) {
        start = std::vector<size_t>(input_dims.size());
      }
      if (end.empty()) {
        end = input_dims;
      }

      double input_sum = 0, decomp_sum = 0;
      const size_t n = compat::transform_reduce(
          std::begin(start), std::end(start),
          std::begin(end),
          size_t{0},
          [](size_t start, size_t stop){ return stop-start; },
          compat::plus<>{}
          );
      switch (input_dims.size()) {
        case 1:
          for (size_t i = start[0]; i < end[0]; ++i) {
            input_sum += input_begin[i];
            decomp_sum += decomp_begin[i];
          }
          break;
        case 2:
          {
          auto stride = input_dims[1];
          for (size_t i = start[0]; i < end[0]; ++i) {
            for (size_t j = start[1]; j < end[1]; ++j) {
              auto idx = j + stride * i;
              input_sum += input_begin[idx];
              decomp_sum += decomp_begin[idx];
            }
          }
          }
          break;
        case 3:
          {
          const auto stride = input_dims[2];
          const auto stride2 = input_dims[1] * input_dims[2];
          for (size_t i = start[0]; i < end[0]; ++i) {
            for (size_t j = start[1]; j < end[1]; ++j) {
              for (size_t k = start[2]; k < end[2]; ++k) {
                auto idx =k + j*stride + i*stride2;
                input_sum += input_begin[idx];
                decomp_sum += decomp_begin[idx];
              }
            }
          }
          }
          break;
        default:
          throw std::runtime_error("region of interest currently only supports dims 1-3");
      }
      m.input_avg = input_sum/static_cast<double>(n);
      m.input_sum = input_sum;
      m.decomp_avg = decomp_sum/static_cast<double>(n);
      m.decomp_sum = decomp_sum;
      return m;
    }

    std::vector<size_t> const input_dims, decomp_dims;
    std::vector<size_t> start,end;
  };
}

class region_of_interest_plugin : public libpressio_metrics_plugin
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
    err_metrics = pressio_data_for_each<region_of_interest::region_of_interest_metrics>( input_data, *output,
        region_of_interest::compute_metrics{input_data.dimensions(), output->dimensions(), start.to_vector<size_t>(), end.to_vector<size_t>()});
  }

  struct pressio_options get_metrics_results() const override
  {
    pressio_options opt;
    set(opt, "region_of_interest:input_average", err_metrics.input_avg);
    set(opt, "region_of_interest:input_sum", err_metrics.input_sum);
    set(opt, "region_of_interest:decomp_average", err_metrics.decomp_avg);
    set(opt, "region_of_interest:decomp_sum", err_metrics.decomp_sum);
    return opt;
  }

  pressio_options get_configuration() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "unstable");
    return opts;
  }
  pressio_options get_options() const override {
    pressio_options opts;
    set(opts, "region_of_interest:start", start);
    set(opts, "region_of_interest:end", end);
    return opts;
  }

  int set_options(pressio_options const& opts) override {
    get(opts, "region_of_interest:start", &start);
    get(opts, "region_of_interest:end", &end);
    return 0;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<region_of_interest_plugin>(*this);
  }

  const char* prefix() const override {
    return "region_of_interest";
  }

private:
  pressio_data input_data = pressio_data::empty(pressio_byte_dtype, {});
  pressio_data start = pressio_data::empty(pressio_uint64_dtype, {}),
               end = pressio_data::empty(pressio_uint64_dtype, {});
  region_of_interest::region_of_interest_metrics err_metrics;
};

static pressio_register metrics_region_of_interest_plugin(metrics_plugins(), "region_of_interest", []() {
  return compat::make_unique<region_of_interest_plugin>();
});
