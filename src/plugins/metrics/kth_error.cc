#include <algorithm>
#include <cmath>
#include "pressio_data.h"
#include "pressio_options.h"
#include "pressio_compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"
#include "std_compat/algorithm.h"

namespace libpressio {
  namespace kth_error {
  struct kth_error{
    template <class ForwardIt1, class ForwardIt2>
    double operator()(ForwardIt1 input_begin, ForwardIt1 input_end,
                             ForwardIt2 decomp_begin, ForwardIt2)
    {
      using value_type = typename std::iterator_traits<ForwardIt1>::value_type;

      std::vector<double> errors(std::distance(input_begin, input_end));
      std::transform(input_begin, input_end, decomp_begin, errors.begin(), [](value_type i, value_type d){
          return std::fabs(i -d);
      });
      auto kth = size_t(static_cast<double>(errors.size()) * k);
      compat::nth_element(errors.begin(), std::next(errors.begin(), kth), errors.end());

      return errors[kth];
    }

    double k;
  };


class kth_error_plugin : public libpressio_metrics_plugin
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
    this->error = pressio_data_for_each<double>(input_data, *output, kth_error{k});
    return 0;
  }

  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    return opts;
  }

  struct pressio_options get_documentation_impl() const override
  {
    pressio_options opt;
    set(opt, "pressio:description", "computes the kth order statistic");
    set(opt, "kth_error:k", "the k order, as a value between 0.0 and 1.0");
    set(opt, "kth_error:kth_error", "the kth order error");
    return opt;
  }

  pressio_options get_metrics_results(pressio_options const &) override
  {
    pressio_options opt;
    set(opt, "kth_error:kth_error", error);
    return opt;
  }

  int set_options(struct pressio_options const& opts) override
  {
    pressio_options opt;
    double tmp_k;
    if(get(opts, "kth_error:k", &tmp_k) == pressio_options_key_set) {
      if(tmp_k >= 0 && tmp_k <= 1.0) {
        k = tmp_k;
      } else {
        return 1;
      }
    }
    return 0;
  }

  pressio_options get_options() const override
  {
    pressio_options opts;
    opts.set("kth_error:k", k);
    return opts;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<kth_error_plugin>(*this);
  }

  const char* prefix() const override {
    return "kth_error";
  }

private:
  pressio_data input_data = pressio_data::empty(pressio_byte_dtype, {});
  compat::optional<double> error;
  double k = .5;
};

static pressio_register metrics_kth_error_plugin(metrics_plugins(), "kth_error", []() {
  return compat::make_unique<kth_error_plugin>();
});
}
}
