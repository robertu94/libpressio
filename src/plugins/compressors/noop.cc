/**
 * a dummy no-op compressor for use in testing and facilitating querying parameters
 */
#include <memory>

#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"

class noop_compressor_plugin: public libpressio_compressor_plugin {
  public:

  struct pressio_options get_configuration_impl() const override {
    return {};
  }

  struct pressio_options get_options_impl() const override {
    return {};
  }

  int set_options_impl(struct pressio_options const&) override {
    return 0;
  }

  int compress_impl(const pressio_data *input, struct pressio_data* output) override {
    *output = pressio_data::clone(*input);
    return 0;
  }

  int decompress_impl(const pressio_data *input, struct pressio_data* output) override {
    *output = pressio_data::clone(*input);
    return 0;
  }

  int major_version() const override {
    return 0;
  }
  int minor_version() const override {
    return 0;
  }
  int patch_version() const override {
    return 0;
  }

  const char* version() const override {
    return "noop 0.0.0.0"; 
  }

  const char* prefix() const override {
    return "noop";
  }
  std::shared_ptr<libpressio_compressor_plugin> clone() override{
    return std::make_unique<noop_compressor_plugin>(*this);
  }
};

static pressio_register X(compressor_plugins(), "noop", [](){ return compat::make_unique<noop_compressor_plugin>();});

