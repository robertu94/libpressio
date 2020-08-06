#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/compat/memory.h"

class printer_plugin : public libpressio_metrics_plugin {
public:

  void begin_check_options(struct pressio_options const* ) override {
    log.emplace_back("begin_check_options");
  }

  void end_check_options(struct pressio_options const*, int ) override {
    log.emplace_back("end_check_options");
  }

  void begin_get_options() override {
    log.emplace_back("begin_get_options");
  }

  void end_get_options(struct pressio_options const* ) override {
    log.emplace_back("end_get_options");
  }

  void begin_get_configuration() override {
    log.emplace_back("begin_get_configuration");
  }

  void end_get_configuration(struct pressio_options const& ) override {
    log.emplace_back("end_get_configuration");
  }


  void begin_set_options(struct pressio_options const& ) override {
    log.emplace_back("begin_set_options");
  }

  void end_set_options(struct pressio_options const& , int ) override {
    log.emplace_back("end_set_options");
  }

  void begin_compress(const struct pressio_data * , struct pressio_data const * ) override {
    log.emplace_back("begin_compress");
  }

  void end_compress(struct pressio_data const* , pressio_data const * , int ) override {
    log.emplace_back("end_compress");
  }

  void begin_decompress(struct pressio_data const* , pressio_data const* ) override {
    log.emplace_back("begin_decompress");
  }

  void end_decompress(struct pressio_data const* , pressio_data const* , int ) override {
    log.emplace_back("end_decompress");
  }

  virtual void begin_compress_many(compat::span<const pressio_data* const> const&,
                                   compat::span<const pressio_data* const> const& ) override {
    log.emplace_back("begin_compress_many");
  }

  void end_compress_many(compat::span<const pressio_data* const> const&,
                                   compat::span<const pressio_data* const> const&, int) override {
    log.emplace_back("end_compress_many");
  }

  virtual void begin_decompress_many(compat::span<const pressio_data* const> const&,
                                   compat::span<const pressio_data* const> const&) override {
    log.emplace_back("begin_decompress_many");
  }

  virtual void end_decompress_many(compat::span<const pressio_data* const> const&,
                                   compat::span<const pressio_data* const> const&, int ) override {
    log.emplace_back("end_decompress_many");
  }


  struct pressio_options get_metrics_results() const override {
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
