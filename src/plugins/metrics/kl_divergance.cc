#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <iterator>
#include "pressio_data.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/compat/memory.h"

namespace kl_divergence{
  struct kl_metrics {
    double p_q=0;
    double q_p=0;
  };

  struct compute_metrics{
    template <class RandomIt1, class RandomIt2>
    kl_metrics operator()(RandomIt1 input_begin, RandomIt1 input_end, RandomIt2 decomp_begin,
                          RandomIt2 decomp_end) {
      using value_type = typename std::iterator_traits<RandomIt1>::value_type;
      kl_metrics m;

      size_t p_size=std::distance(input_begin, input_end), q_size=std::distance(decomp_begin, decomp_end);
      std::unordered_map<value_type, size_t> p_counts, q_counts;
      std::unordered_set<value_type> X;
      p_counts.reserve(p_size);
      q_counts.reserve(q_size);
      X.reserve(p_size + q_size);
      std::for_each( input_begin, input_end, [&p_counts,&X](value_type p) { p_counts[p] += 1; X.insert(p);});
      std::for_each( decomp_begin, decomp_end, [&q_counts,&X](value_type q) { q_counts[q] += 1; X.insert(q);});

      for (auto const& x : X) {
        m.p_q += p_counts[x] / p_size * std::log((p_counts[x] * q_size)/ (q_counts[x]* p_size));
        m.q_p += q_counts[x] / q_size * std::log((q_counts[x] * p_size)/ (p_counts[x]* q_size));
        
      }

      return m;
    }
  };
}

class kl_divergance_plugin : public libpressio_metrics_plugin {

public:
  void begin_compress(const struct pressio_data* input,
                      struct pressio_data const*) override
  {
    input_data = pressio_data::clone(*input);
  }
  void end_decompress(struct pressio_data const*,
                      struct pressio_data const* output, int) override
  {
    err_metrics = pressio_data_for_each<kl_divergence::kl_metrics>(input_data, *output,
                                                       kl_divergence::compute_metrics{});
  }

  struct pressio_options get_metrics_results() const override
  {
    pressio_options opt;
    if (err_metrics) {
      set(opt, "kl_divergence:q_p", (*err_metrics).q_p);
      set(opt, "kl_divergence:p_q", (*err_metrics).p_q);
    } else {
      set_type(opt, "kl_divergence:q_p", pressio_option_double_type);
      set_type(opt, "kl_divergence:p_q", pressio_option_double_type);
    }
    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<kl_divergance_plugin>(*this);
  }

  const char* prefix() const override {
    return "kl_divergence";
  }

private:
  pressio_data input_data = pressio_data::empty(pressio_byte_dtype, {});
  compat::optional<kl_divergence::kl_metrics> err_metrics;
};

static pressio_register metrics_kl_divergance_plugin(metrics_plugins(), "kl_divergence",
                          []() { return compat::make_unique<kl_divergance_plugin>(); });
