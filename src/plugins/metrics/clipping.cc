#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "std_compat/memory.h"

namespace libpressio { namespace clipping_metrics_ns {

class clipping_plugin : public libpressio_metrics_plugin {
  public:
    int end_compress_impl(struct pressio_data const* input, pressio_data const*, int) override {
      if(!input || !input->has_data()) return 0;
      this->input = pressio_data::clone(domain_manager().make_readable(domain_plugins().build("malloc"), *input));
      return 0;
    }

    struct clipping_op {
      template <class T, class U>
      uint64_t operator()(T const* in_begin, T const* in_end, U const* out_begin, U const* out_end) {
        const size_t n = std::min(
            std::distance(in_begin, in_end),
            std::distance(out_begin, out_end)
            );
        size_t count = 0;
        for (size_t i = 0; i < n; ++i) {
          double diff = std::abs(static_cast<double>(in_begin[i]) - static_cast<double>(out_begin[i]));
          if (diff > config->abs_bound) {
            count++;
          }
        }
        return count;
      }
      clipping_plugin const* config;
    };

    int end_decompress_impl(struct pressio_data const* , pressio_data const* output, int) override {
      if(!output || !output->has_data() || !this->input.has_data()) return 0;
      clips = pressio_data_for_each<uint64_t>(input, domain_manager().make_readable(domain_plugins().build("malloc"), *output), clipping_op{this});
      return 0;
    }

  struct pressio_options get_options() const override {
    pressio_options options;
    set(options, "pressio:abs", abs_bound);
    set(options, "clipping:abs", abs_bound);
    return options;
  }

  int set_options(pressio_options const& options) override {
    get(options, "pressio:abs", &abs_bound);
    get(options, "clipping:abs", &abs_bound);
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
    set(opt, "pressio:description", "measures the number of values that exceed an absolute threshold");
    set(opt, "clipping:abs", "threshold for error");
    set(opt, "pressio:abs", "threshold for error");
    set(opt, "clipping:clips", "the number of clips that occur");
    return opt;
  }

  pressio_options get_metrics_results(pressio_options const &) override {
    pressio_options opt;
    set(opt, "clipping:clips", clips);
    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<clipping_plugin>(*this);
  }
  const char* prefix() const override {
    return "clipping";
  }

  private:
  compat::optional<uint64_t> clips = 0;
  double abs_bound = 1e-4;
  pressio_data input;
};

static pressio_register metrics_clipping_plugin(metrics_plugins(), "clipping", [](){ return compat::make_unique<clipping_plugin>(); });
}}
