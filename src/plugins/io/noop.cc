#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "libpressio_ext/io/posix.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/io.h"

struct noop_io : public libpressio_io_plugin {
  virtual struct pressio_data* read_impl(struct pressio_data* data) override {
    if(data != nullptr) pressio_data_free(data);
    return nullptr;
  }

  virtual int write_impl(struct pressio_data const*) override{
    return 0;
  }
  virtual struct pressio_options get_configuration_impl() const override{
    return {
      {"pressio:thread_safe",  static_cast<int>(pressio_thread_safety_multiple)}
    };
  }

  virtual int set_options_impl(struct pressio_options const&) override{
    return 0;
  }
  virtual struct pressio_options get_options_impl() const override{
    return {};
  }

  int patch_version() const override{ 
    return 1;
  }
  virtual const char* version() const override{
    return "0.0.1";
  }
  const char* prefix() const override {
    return "noop";
  }
  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<noop_io>(*this);
  }

  private:
};

static pressio_register X(io_plugins(), "noop", [](){ return compat::make_unique<noop_io>(); });

