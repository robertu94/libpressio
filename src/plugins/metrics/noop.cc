#include "pressio_options.h"
#include "pressio_compressor.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"

class noop_metrics_plugin : public libpressio_metrics_plugin {
public:
  pressio_options get_metrics_results(pressio_options const &) const override {
    return pressio_options();
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<noop_metrics_plugin>(*this);
  }

  struct pressio_options get_configuration() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_multiple));
    return opts;
  }

  pressio_options get_documentation_impl() const override {
    pressio_options opts;
    set(opts, "pressio:description", "a metric that does nothing");
    return opts;
  }

  const char* prefix() const override {
    return "noop";
  }
};

static pressio_register metrics_noop_plugin(metrics_plugins(), "noop", [](){ return compat::make_unique<noop_metrics_plugin>(); });
