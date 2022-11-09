#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/printers.h"
#include "std_compat/memory.h"
#include <iostream>

namespace libpressio { namespace print_options_metrics_ns {

class print_options_plugin : public libpressio_metrics_plugin {
  public:

  int begin_set_options_impl(struct pressio_options const& options) override {
    std::cout << options;
    return 0;
  }

  
  struct pressio_options get_configuration() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", "prints options passed to set_options in a human readable form");
    return opt;
  }

  pressio_options get_metrics_results(pressio_options const &) override {
    pressio_options opt;
    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<print_options_plugin>(*this);
  }
  const char* prefix() const override {
    return "print_options";
  }

  private:

};

static pressio_register metrics_print_options_plugin(metrics_plugins(), "print_options", [](){ return compat::make_unique<print_options_plugin>(); });
}}

