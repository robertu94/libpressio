#include <sys/stat.h>
#include <unistd.h>
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "libpressio_ext/io/posix.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/io.h"
#include "std_compat/memory.h"

namespace libpressio { namespace io { namespace noop_ns {
struct noop_io : public libpressio_io_plugin {
  virtual struct pressio_data* read_impl(struct pressio_data*) override {
    return nullptr;
  }

  virtual int write_impl(struct pressio_data const*) override{
    return 0;
  }
  virtual struct pressio_options get_configuration_impl() const override{
    pressio_options opts;
    set(opts, "pressio:thread_safe",  pressio_thread_safety_multiple);
    set(opts, "pressio:stability", "stable");
    return opts;
  }

  virtual int set_options_impl(struct pressio_options const&) override{
    return 0;
  }
  virtual struct pressio_options get_options_impl() const override{
    return {};
  }
  virtual struct pressio_options get_documentation_impl() const override{
    pressio_options opts;
    set(opts, "pressio:description", "a writer which preforms a noop");
    set(opts, "pressio:stability", "stable");

    return opts;
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

pressio_register registration(io_plugins(), "noop", [](){ return compat::make_unique<noop_io>(); });

} } }
