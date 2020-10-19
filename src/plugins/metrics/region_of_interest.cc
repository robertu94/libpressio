#include <cmath>
#include <iterator>
#include <stdexcept>
#include "pressio_data.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/compat/memory.h"

namespace region_of_interest {
  struct region_of_interest_metrics {
    compat::optional<double> avg;
    compat::optional<double> sum;
  };

  struct compute_metrics{
    template <class RandomIt1, class RandomIt2>
    region_of_interest_metrics operator()(RandomIt1 input_begin, RandomIt1 ,
                             RandomIt2 , RandomIt2 )
    {
      region_of_interest_metrics m;
      if (start.empty()) {
        start = std::vector<uint64_t>(input_dims.size());
      }
      if (end.empty()) {
        end = input_dims;
      }

      double sum = 0;
      size_t n = 0;
      switch (input_dims.size()) {
        case 1:
          for (uint64_t i = start[0]; i < end[0]; ++i) {
            sum += input_begin[i];
            n++;
          }
          break;
        case 2:
          {
          auto stride = input_dims[1];
          for (uint64_t i = start[0]; i < end[0]; ++i) {
            for (uint64_t j = start[1]; j < end[1]; ++j) {
              auto idx = j + stride * i;
              sum += input_begin[idx];
              n++;
            }
          }
          }
          break;
        case 3:
          {
          const auto stride = input_dims[2];
          const auto stride2 = input_dims[1] * input_dims[2];
          for (uint64_t i = start[0]; i < end[0]; ++i) {
            for (uint64_t j = start[1]; j < end[1]; ++j) {
              for (uint64_t k = start[2]; k < end[2]; ++k) {
                auto idx =k + j*stride + i*stride2;
                sum += input_begin[idx];
                n++;
              }
            }
          }
          }
          break;
        default:
          throw std::runtime_error("region of interest currently only supports dims 1-3");
      }
      m.avg = sum/static_cast<double>(n);
      m.sum = sum;
      return m;
    }

    std::vector<size_t> const input_dims, decomp_dims;
    std::vector<uint64_t> start,end;
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
        region_of_interest::compute_metrics{input_data.dimensions(), output->dimensions(), start.to_vector<uint64_t>(), end.to_vector<uint64_t>()});
  }

  struct pressio_options get_metrics_results() const override
  {
    pressio_options opt;
    set(opt, "region_of_interest:average", err_metrics.avg);
    set(opt, "region_of_interest:sum", err_metrics.sum);
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
