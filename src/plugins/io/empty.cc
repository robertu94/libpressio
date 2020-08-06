#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <sstream>
#include <fstream>
#include <memory>
#include <algorithm>
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "libpressio_ext/io/posix.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/io.h"
#include "libpressio_ext/compat/memory.h"


namespace {
  struct zero {
    template <class T>
    int operator()(T* begin, T* end) {
      std::fill(begin, end, T{0});
      return 0;
    }
  };
}

struct empty_io : public libpressio_io_plugin
{
  virtual struct pressio_data* read_impl(struct pressio_data* data) override {
    auto empty = new pressio_data(pressio_data::owning(data->dtype(), data->dimensions()));
    pressio_data_for_each<int>(*empty, zero{});
    return empty;
  }

  virtual int write_impl(struct pressio_data const*) override{
    //intensional no-op
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
    return pressio_options{};
  }

  int patch_version() const override{ 
    return 1;
  }

  virtual const char* version() const override{
    return "0.0.1";
  }

  const char* prefix() const override {
    return "empty";
  }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<empty_io>(*this);
  }

  private:
};

static pressio_register io_empty_plugin(io_plugins(), "empty",
                          []() { return compat::make_unique<empty_io>(); });

