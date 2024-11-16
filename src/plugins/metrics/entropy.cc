#include <cmath>
#include "pressio_data.h"
#include "pressio_options.h"
#include "pressio_compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "std_compat/memory.h"
#include <vector>
#include <map>
#include <cmath>

namespace libpressio {
namespace entropy {
  struct compute_metrics{
    template <class ForwardIt1>
    double operator()(ForwardIt1 input_begin, ForwardIt1 input_end)
    {
      size_t total = 0;
      std::map<typename std::iterator_traits<ForwardIt1>::value_type, size_t> values;
        ;
      for(auto i = input_begin; i!=input_end; ++i) {
        auto&& it = values.find(*i);
        if(it != values.end()) {
          it->second += 1;
        } else {
          values.emplace(*i, 1);
        }
        total++;
      }
      double entropy = 0;
      for (auto const& i : values) {
        double v = static_cast<double>(i.second) / static_cast<double>(total);
        entropy +=  v * std::log2(v);
      }
      return -entropy;
    }
  };

class entropy_plugin : public libpressio_metrics_plugin
{

public:
  int begin_compress_impl(const struct pressio_data* input, struct pressio_data const*) override
  {
      if(!input || !input->has_data()) return 0;
    input_entropy = pressio_data_for_each<double>(domain_manager().make_readable(domain_plugins().build("malloc"), *input), entropy::compute_metrics{});
    return 0;
  }
  int end_decompress_impl(struct pressio_data const*, struct pressio_data const* output, int) override
  {
      if(!output || !output->has_data()) return 0;
    dec_entropy = pressio_data_for_each<double>(domain_manager().make_readable(domain_plugins().build("malloc"), *output), entropy::compute_metrics{});
    return 0;
  }

  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(opts, "predictors:requires_decompress", std::vector<std::string>{"entropy:decompressed"});
    set(opts, "predictors:data", std::vector<std::string>{"entropy:input"});
    set(opts, "predictors:error_dependent", std::vector<std::string>{"entropy:decompressed"});
    return opts;
  }


  pressio_options get_documentation_impl() const override {
    pressio_options opts;
    set(opts, "pressio:description", "computes the entropy of the input data and output data");
    set(opts, "entropy:input", "the entropy of the input data (shannon)");
    set(opts, "entropy:decompressed", "the entropy of the decompressed data (shannon)");
    return opts;
  }

  pressio_options get_metrics_results(pressio_options const &)  override
  {
    pressio_options opt;
    set(opt, "entropy:input", input_entropy);
    set(opt, "entropy:decompressed", dec_entropy);
    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<entropy_plugin>(*this);
  }

  const char* prefix() const override {
    return "entropy";
  }

private:
  pressio_data input_data = pressio_data::empty(pressio_byte_dtype, {});
  compat::optional<double> input_entropy;
  compat::optional<double> dec_entropy;
};

static pressio_register metrics_entropy_plugin(metrics_plugins(), "entropy", []() {
  return compat::make_unique<entropy_plugin>();
});
}}
