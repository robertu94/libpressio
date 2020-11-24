#include <chrono>
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"

class noop_metrics_plugin : public libpressio_metrics_plugin {
public:

  struct pressio_options get_metrics_results() const override {
    return pressio_options();
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<noop_metrics_plugin>(*this);
  }

  const char* prefix() const override {
    return "noop";
  }
};

static pressio_register metrics_noop_plugin(metrics_plugins(), "noop", [](){ return compat::make_unique<noop_metrics_plugin>(); });
