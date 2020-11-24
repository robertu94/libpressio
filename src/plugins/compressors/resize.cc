#include <vector>
#include <memory>
#include <random>
#include <numeric>
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "pressio_options.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "std_compat/memory.h"


class resize_meta_compressor_plugin : public libpressio_compressor_plugin
{
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "resize:compressor", compressor_id, compressor);
    set(options, "resize:compressed_dims", pressio_data(compressed_dims.begin(), compressed_dims.end()));
    set(options, "resize:decompressed_dims", pressio_data(decompressed_dims.begin(), decompressed_dims.end()));
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", static_cast<int>(pressio_thread_safety_multiple));
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override
  {
    get_meta(options, "resize:compressor", compressor_plugins(), compressor_id, compressor);
    pressio_data tmp;
    if(get(options, "resize:compressed_dims", &tmp) == pressio_options_key_set) {
      compressed_dims = tmp.to_vector<size_t>();
    }
    if(get(options, "resize:decompressed_dims", &tmp) == pressio_options_key_set) {
      decompressed_dims = tmp.to_vector<size_t>();
    }
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    auto tmp = pressio_data::clone(*input);
    tmp.reshape(compressed_dims);
    return compressor->compress(&tmp, output);
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    auto ret = compressor->decompress(input, output);
    output->reshape(compressed_dims);
    return ret;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }

  const char* version() const override { return "0.0.1"; }

  const char* prefix() const override { return "resize"; }

  void set_name_impl(std::string const& name) override {
    compressor->set_name(name + "/" + compressor->prefix());
  }

  pressio_options get_metrics_results_impl() const override {
    return compressor->get_metrics_results();
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<resize_meta_compressor_plugin>(*this);
  }

private:
  std::vector<size_t> compressed_dims;
  std::vector<size_t> decompressed_dims;
  pressio_compressor compressor = compressor_plugins().build("noop");
  std::string compressor_id = "noop";
};

static pressio_register compressor_resize_plugin(compressor_plugins(), "resize", [](){ return compat::make_unique<resize_meta_compressor_plugin>(); });

