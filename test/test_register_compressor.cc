#include <cmath>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <sz.h>

#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/compat/memory.h"
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
  //compress and decompress
  int compress_impl(pressio_data const* input, pressio_data* output) override {
    pressio_data log_input = pressio_data_for_each<pressio_data>(*input, log_fn{input});
    return check_error(compressor.plugin->compress(&log_input, output));
  }

  int decompress_impl(pressio_data const* input, pressio_data* output) override {
    int rc =  compressor.plugin->decompress(input, output);
    *output = pressio_data_for_each<pressio_data>(*output, exp_fn{output});
    return check_error(rc);
  }

  //getting and setting options/configuration
  pressio_options get_options_impl() const override {
    auto options =  compressor.plugin->get_options();
    set(options, "log:compressor", compressor_id);
    return options;
  }
  int set_options_impl(pressio_options const& options) override {
    int rc = check_error(compressor.plugin->set_options(options));
    std::string tmp;
    if(get(options, "log:compressor", &tmp) == pressio_options_key_set) {
      pressio library;
      if(auto tmp_compressor = library.get_compressor(tmp)) {
        compressor = std::move(tmp_compressor);
        compressor_id = std::move(tmp);
      } else {
        return set_error(library.err_code(), library.err_msg());
      }
    }
    return rc;
  }
  pressio_options get_configuration_impl() const override {
    return compressor.plugin->get_configuration();
  }
  int check_options_impl(pressio_options const& options) override {
    return check_error(compressor.plugin->check_options(options));
  }

  //getting version information
  const char* prefix() const override {
    return "log";
  }
  const char* version() const override {
    return "0.1.0";
  }
  int major_version() const override {
    return 0;
  }
  int minor_version() const override {
    return 1;
  }
  int patch_version() const override {
    return 0;
  }
  std::shared_ptr<libpressio_compressor_plugin> clone() override {
    return compat::make_unique<log_transform>(*this);
  }

  private:
  void set_name_impl(std::string const& new_name) override {
    compressor->set_name(new_name + "/" + compressor->prefix());
  }

  int check_error(int rc) { 
    if(rc) {
      set_error(compressor->error_code(), compressor->error_msg());
    }
    return rc;
  }
  pressio_compressor compressor = compressor_plugins().build("noop");
  std::string compressor_id = "noop";
};

static pressio_register X(compressor_plugins(), "log", [](){ return compat::make_unique<log_transform>();});

TEST(ExternalPlugin, TestLogCompressor) {
  pressio library;

  auto log_compressor = library.get_compressor("log");
  auto options = log_compressor->get_options();
  options.set("log:compressor", "sz");
  options.set("sz:error_bound_mode", ABS);
  options.set("sz:abs_err_bound", 0.5);

  if(log_compressor->check_options(options)) {
    FAIL() << log_compressor->error_msg() << std::endl;
  }

  if(log_compressor->set_options(options)) {
    FAIL() << log_compressor->error_msg() << std::endl;
  }

  double* rawinput_data = make_input_data();
  //providing a smaller than expected buffer to save time during testing
  std::vector<size_t> dims{30,30,30};

  auto input = pressio_data::move(pressio_double_dtype, rawinput_data, dims, pressio_data_libc_free_fn, nullptr);

  auto compressed = pressio_data::empty(pressio_byte_dtype, {});

  auto decompressed = pressio_data::empty(pressio_double_dtype, dims);

  if(log_compressor->compress(&input, &compressed)) {
    FAIL() << library.err_msg() << std::endl;
  }

  if(log_compressor->decompress(&compressed, &decompressed)) {
    FAIL() << library.err_msg() << std::endl;
  }
}

TEST(ExternalPlugin, TestNames) {
  pressio library;
  auto log_compressor = library.get_compressor("log");
  log_compressor->set_options({{"log:compressor", "sample"}});
  log_compressor->set_name("log_example");

  auto options = log_compressor->get_options();
  std::vector<std::string> actual_names;
  std::transform(
      std::begin(options),
      std::end(options),
      std::back_inserter(actual_names),
      [](std::pair<std::string, pressio_option> const& item){
        return item.first;
      }
      );
  std::vector<std::string> expected_names{
    "/log_example:log:compressor",
    "/log_example:log:metric",
    "/log_example/sample:sample:seed",
    "/log_example/sample:sample:rate",
    "/log_example/sample:sample:mode",
    "/log_example/sample:sample:metric"
  };
  EXPECT_THAT(actual_names, ::testing::IsSupersetOf(expected_names));


  log_compressor->set_name("log_example2");
  std::vector<std::string> actual_names2;
  std::vector<std::string> expected_names2{
    "/log_example2:log:compressor",
    "/log_example2/sample:sample:seed",
    "/log_example2/sample:sample:rate",
    "/log_example2/sample:sample:mode"
  };
  {
    auto options = log_compressor->get_options();
    std::transform(
        std::begin(options),
        std::end(options),
        std::back_inserter(actual_names2),
        [](std::pair<std::string, pressio_option> const& item){
          return item.first;
        }
        );
  }
  EXPECT_THAT(actual_names2, ::testing::IsSupersetOf(expected_names2));
}

