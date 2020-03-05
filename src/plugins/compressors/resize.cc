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

namespace {
  /**
   * converts a pressio_data structure to a std::vector of the template type
   *
   * \param[in] data the data to convert
   * \returns a new vector
   */
  template <class T>
  std::vector<T> pressio_data_to_vector(pressio_data const& data) {
    return std::vector<T>(static_cast<T*>(data.data()), static_cast<T*>(data.data()) + data.num_elements());
  }


  /**
   * converts a std::vector of template type to a pressio_data structure
   *
   * \param[in] vec the data to convert
   * \returns a pressio_data structure
   */
  template <class T>
  pressio_data vector_to_owning_pressio_data(std::vector<T> const& vec) {
    return pressio_data::copy(pressio_dtype_from_type<T>(), vec.data(), {vec.size()});
  }
}

class resize_meta_compressor_plugin : public libpressio_compressor_plugin
{
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    options.set("resize:compressor", compressor_id);
    options.set("resize:compressed_dims", vector_to_owning_pressio_data(compressed_dims));
    options.set("resize:decompressed_dims", vector_to_owning_pressio_data(decompressed_dims));
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    options.set("pressio:thread_safe",
                static_cast<int>(pressio_thread_safety_multiple));
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override
  {
    if(options.get("resize:compressor", &compressor_id) == pressio_options_key_set) {
      pressio library;
      compressor = library.get_compressor(compressor_id);
    }
    pressio_data tmp;
    if(options.get("resize:compressed_dims", &tmp) == pressio_options_key_set) {
      compressed_dims = pressio_data_to_vector<size_t>(tmp);
    }
    if(options.get("resize:decompressed_dims", &tmp) == pressio_options_key_set) {
      decompressed_dims = pressio_data_to_vector<size_t>(tmp);
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

static pressio_register X(compressor_plugins(), "resize", [](){ return compat::make_unique<resize_meta_compressor_plugin>(); });

