#include <numeric>
#include <vector>
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/io.h"
#include "std_compat/memory.h"

namespace libpressio { namespace iota {
namespace iota_plugin {
  struct apply_iota {
    template <class T> int operator()(T* begin, T* end) {
      std::iota(begin, end, 0);
      return 0;
    }
  };
}

struct iota_io : public libpressio_io_plugin {
  virtual struct pressio_data* read_impl(struct pressio_data* buf) override {
    if(!buf) {
      set_error(1, "input dimensions are required");
      return nullptr;
    } else {
      auto* out = new pressio_data();
      if(buf->has_data()) {
        *out = std::move(*buf);
      } else {
        *out = pressio_data::owning(
            buf->dtype(),
            buf->dimensions()
            );
      }
      //apply iota
      pressio_data_for_each<int>(*out, iota_plugin::apply_iota{});

      return out;
    }
  }

  virtual int write_impl(struct pressio_data const*) override{
    return set_error(1, "write not supported");
  }

  virtual struct pressio_options get_configuration_impl() const override{
    pressio_options opts; 
    set(opts, "pressio:thread_safe",  pressio_thread_safety_multiple);
    set(opts, "pressio:stability",  "stable");
    return opts;
  }

  virtual struct pressio_options get_documentation_impl() const override{
    pressio_options opts;
    set(opts, "pressio:description", "read in a block that counts from 1 to N");
    set(opts, "pressio:stability", "stable");
    return opts;
  }
  virtual int set_options_impl(struct pressio_options const&) override{
    return 0;
  }
  virtual struct pressio_options get_options_impl() const override{
    pressio_options opts;
    return opts;
  }

  int patch_version() const override{ 
    return 1;
  }
  virtual const char* version() const override{
    return "0.0.1";
  }
  const char* prefix() const override {
    return "iota";
  }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<iota_io>(*this);
  }

  private:
  std::string path;
};

static pressio_register io_posix_plugin(io_plugins(), "iota", [](){ return compat::make_unique<iota_io>(); });
} }
