#include "pressio_options.h"
#include "pressio_compressor.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"

namespace libpressio {namespace printer {
class printer_plugin : public libpressio_metrics_plugin {
public:
  int begin_check_options_impl(struct pressio_options const* ) override {
    log.emplace_back("begin_check_options");
    return 0;
  }

  int end_check_options_impl(struct pressio_options const*, int ) override {
    log.emplace_back("end_check_options");
    return 0;
  }

  int begin_get_options_impl() override {
    log.emplace_back("begin_get_options");
    return 0;
  }

  int end_get_options_impl(struct pressio_options const* ) override {
    log.emplace_back("end_get_options");
    return 0;
  }

  int begin_get_configuration_impl() override {
    log.emplace_back("begin_get_configuration");
    return 0;
  }

  int end_get_configuration_impl(struct pressio_options const& ) override {
    log.emplace_back("end_get_configuration");
    return 0;
  }

  int begin_set_options_impl(struct pressio_options const& ) override {
    log.emplace_back("begin_set_options");
    return 0;
  }

  int end_set_options_impl(struct pressio_options const& , int ) override {
    log.emplace_back("end_set_options");
    return 0;
  }

  int begin_compress_impl(const struct pressio_data * , struct pressio_data const * ) override {
    log.emplace_back("begin_compress");
    return 0;
  }

  int end_compress_impl(struct pressio_data const* , pressio_data const * , int ) override {
    log.emplace_back("end_compress");
    return 0;
  }

  int begin_decompress_impl(struct pressio_data const* , pressio_data const* ) override {
    log.emplace_back("begin_decompress");
    return 0;
  }

  int end_decompress_impl(struct pressio_data const* , pressio_data const* , int ) override {
    log.emplace_back("end_decompress");
    return 0;
  }

  int begin_compress_many_impl(compat::span<const pressio_data* const> const&,
                                   compat::span<const pressio_data* const> const& ) override {
    log.emplace_back("begin_compress_many");
    return 0;
  }

  int end_compress_many_impl(compat::span<const pressio_data* const> const&,
                                   compat::span<const pressio_data* const> const&, int) override {
    log.emplace_back("end_compress_many");
    return 0;
  }

  int begin_decompress_many_impl(compat::span<const pressio_data* const> const&,
                                   compat::span<const pressio_data* const> const&) override {
    log.emplace_back("begin_decompress_many");
    return 0;
  }

  int end_decompress_many_impl(compat::span<const pressio_data* const> const&,
                                   compat::span<const pressio_data* const> const&, int ) override {
    log.emplace_back("end_decompress_many");
    return 0;
  }


  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(opts, "predictors:requires_decompress", false);
    set(opts, "predictors:invalidate", std::vector<std::string>{});
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", "metric that records the operations preformed");
    set(opt, "printer:log", "log of operations preformed on this metrics object");
    return opt;
  }

  pressio_options get_metrics_results(pressio_options const &) override {
    pressio_options opt;
    set(opt, "printer:log", log);
    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<printer_plugin>(*this);
  }

  const char* prefix() const override {
    return "printer";
  }

  private:
  std::vector<std::string> log;
};

static pressio_register printer_time_plugin(metrics_plugins(), "printer",
                                            []() { return compat::make_unique<printer_plugin>(); });
} }
