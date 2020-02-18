#include <iterator>
#include <cmath>
#include <gtest/gtest.h>
#include <sz/sz.h>

#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/compat/std_compat.h"
#include "make_input_data.h"

namespace {
  struct log_fn{
    template <class T>
    pressio_data operator()(T* begin, T* end) {
        pressio_data log_data = pressio_data::clone(*input);
        auto output_it = reinterpret_cast<T*>(log_data.data());
        std::transform(begin, end, output_it, [](T i){ return std::log(i); });
        return log_data;
    }
    pressio_data const* input;
  };

  struct exp_fn{
    template <class T>
    pressio_data operator()(T* begin, T* end) {
        pressio_data log_data = pressio_data::clone(*output);
        auto output_it = reinterpret_cast<T*>(log_data.data());
        std::transform(begin, end, output_it, [](T i){ return std::exp(i); });
        return log_data;
    }
    pressio_data const* output;
  };
}

class log_transform : public libpressio_compressor_plugin {
  public:
  log_transform(): compressor(nullptr) {}
  log_transform(pressio_compressor&& comp): compressor(std::move(comp)) {}


  //compress and decompress
  int compress_impl(pressio_data const* input, pressio_data* output) override {
    if(!compressor) return invalid_compressor();
    pressio_data log_input = pressio_data_for_each<pressio_data>(*input, log_fn{input});
    return check_error(compressor.plugin->compress(&log_input, output));
  }

  int decompress_impl(pressio_data const* input, pressio_data* output) override {
    if(!compressor) return invalid_compressor();
    int rc =  compressor.plugin->decompress(input, output);
    *output = pressio_data_for_each<pressio_data>(*output, exp_fn{output});
    return check_error(rc);
  }

  //getting and setting options/configuration
  pressio_options get_options_impl() const override {
    auto options =  compressor.plugin->get_options();
    options.set("log:compressor", (void*)&compressor);
    return options;
  }
  int set_options_impl(pressio_options const& options) override {
    if(!compressor) return invalid_compressor();
    int rc = check_error(compressor.plugin->set_options(options));
    void* tmp;
    if(options.get("log:compressor", &tmp) == pressio_options_key_set) {
      compressor = std::move(*(pressio_compressor*)tmp);
    }
    return rc;
  }
  pressio_options get_configuration_impl() const override {
    if(!compressor) return pressio_options();
    return compressor.plugin->get_configuration();
  }
  int check_options_impl(pressio_options const& options) override {
    if(!compressor) return invalid_compressor();
    return check_error(compressor.plugin->check_options(options));
  }

  //getting version information
  const char* prefix() const override {
    return "log";
  }
  const char* version() const override {
    return compressor.plugin->version();
  }
  int major_version() const override {
    return compressor.plugin->major_version();
  }
  int minor_version() const override {
    return compressor.plugin->minor_version();
  }
  int patch_version() const override {
    return compressor.plugin->patch_version();
  }
  std::shared_ptr<libpressio_compressor_plugin> clone() override {
    auto tmp = compat::make_unique<log_transform>();
    tmp->set_error(error_code(), error_msg());
    tmp->compressor = tmp->compressor->clone();
    return tmp;
  }

  private:
  int check_error(int rc) { 
    if(rc) {
      set_error(
          compressor.plugin->error_code(),
          compressor.plugin->error_msg()
          );
    }
    return rc;
  }
  int invalid_compressor() { return set_error(-1, "compressor must be set"); };
  pressio_compressor compressor;
};

static pressio_register X(compressor_plugins(), "log", [](){ return compat::make_unique<log_transform>();});

TEST(ExternalPlugin, TestLogCompressor) {
  pressio library;

  auto sz_compressor = library.get_compressor("sz");
  auto log_compressor = compat::make_unique<log_transform>(std::move(sz_compressor));
  auto options = log_compressor->get_options();
  options.set("sz:error_bound_mode", ABS);
  options.set("sz:abs_err_bound", 0.5);

  if(log_compressor->check_options(options)) {
    std::cerr << log_compressor->error_msg() << std::endl;
    exit(log_compressor->error_code());
  }

  if(log_compressor->set_options(options)) {
    std::cerr << log_compressor->error_msg() << std::endl;
    exit(log_compressor->error_code());
  }

  double* rawinput_data = make_input_data();
  //providing a smaller than expected buffer to save time during testing
  std::vector<size_t> dims{30,30,30};

  auto input = pressio_data::move(pressio_double_dtype, rawinput_data, dims, pressio_data_libc_free_fn, nullptr);

  auto compressed = pressio_data::empty(pressio_byte_dtype, {});

  auto decompressed = pressio_data::empty(pressio_double_dtype, dims);

  if(log_compressor->compress(&input, &compressed)) {
    std::cerr << library.err_msg() << std::endl;
    exit(library.err_code());
  }

  if(log_compressor->decompress(&compressed, &decompressed)) {
    std::cerr << library.err_msg() << std::endl;
    exit(library.err_code());
  }
}

