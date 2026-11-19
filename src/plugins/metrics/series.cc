
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "std_compat/memory.h"

namespace libpressio { namespace metrics { namespace series_metrics_ns {

class series_plugin : public libpressio_metrics_plugin {
  public:
    int end_compress_impl(struct pressio_data const* real_input, pressio_data const* real_output, int rc) override {
      child->end_compress(real_input, real_output, rc);
      return 0;
    }

    int end_decompress_impl(struct pressio_data const* real_input, pressio_data const* real_output, int rc) override {
      child->end_decompress(real_input, real_output, rc);
      return 0;
    }

    int end_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs, int rc) override {
      child->end_compress_many(inputs, outputs, rc);
      return 0;
  }

  int end_decompress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs, int rc) override {
      child->end_decompress_many(inputs, outputs, rc);
    return 0;
  }

  
  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "experimental");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(opts, "predictors:requires_decompress", true);
    set(opts, "predictors:invalidate", std::vector<std::string>{"predictors:error_dependent"});
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", R"()");
    return opt;
  }

  pressio_options get_metrics_results(pressio_options const &) override {
    pressio_options opt;
    opt.copy_from(child->get_metrics_results(opt));
    pressio_data qoi_data;
    if(get(opt, qoi, &qoi_data) == pressio_options_key_set) {
        //compute p90, p99, p999, median, mean, max, wasstinedistance
        //qoi_data
    }

    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<series_plugin>(*this);
  }
  const char* prefix() const override {
    return "series";
  }

  private:
  std::string qoi;
  pressio_metric child;

};

static pressio_register libpressio_metrics_series_plugin(metrics_plugins(), "series", [](){ return compat::make_unique<series_plugin>(); });
}}}

