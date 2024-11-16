#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <iterator>
#include "pressio_data.h"
#include "pressio_options.h"
#include "pressio_compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/domain_manager.h"

namespace libpressio {
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
        m.p_q += static_cast<double>(p_counts[x]) / static_cast<double>(p_size) * std::log((p_counts[x] * q_size)/ (static_cast<double>(q_counts[x])* static_cast<double>(p_size)));
        m.q_p += static_cast<double>(q_counts[x]) / static_cast<double>(q_size) * std::log((q_counts[x] * p_size)/ (static_cast<double>(p_counts[x])* static_cast<double>(q_size)));
        
      }

      return m;
    }
  };

class kl_divergance_plugin : public libpressio_metrics_plugin {

public:
  int begin_compress_impl(const struct pressio_data* input,
                      struct pressio_data const*) override
  {
    if(!input || !input->has_data()) return 0;
    input_data = pressio_data::clone(domain_manager().make_readable(domain_plugins().build("malloc"), *input));
    return 0;
  }
  int end_decompress_impl(struct pressio_data const*,
                      struct pressio_data const* output, int) override
  {
    if(!output || !output->has_data() || !input_data.has_data()) return 0;
    err_metrics = pressio_data_for_each<kl_divergence::kl_metrics>(input_data, domain_manager().make_readable(domain_plugins().build("malloc"), *output),
                                                       kl_divergence::compute_metrics{});
    return 0;
  }

  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(opts, "predictors:requires_decompress", true);
    set(opts, "predictors:invalidate", std::vector<std::string>{"predictors:error_dependent"});
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", "Kullback–Leibler divergence");
    set(opt, "kl_divergence:q_p", "relative entropy of q given p");
    set(opt, "kl_divergence:p_q", "relative entropy of p given q");
    return opt;
  }
  pressio_options get_metrics_results(pressio_options const &) override
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
}
}
