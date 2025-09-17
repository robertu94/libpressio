#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "MGARDx/decompose.hpp"
#include "MGARDx/recompose.hpp"

namespace libpressio { namespace compressor { namespace mgardx_ns {

  struct compress_op {
    template <class T>
    pressio_data operator()(T* begin) {
      size_t compressed_size;
      MGARD::Decomposer<T> decomposer(use_sz);
      auto compressed_data = decomposer.compress(begin, dims, target_level, eb, compressed_size);
      return pressio_data::move(pressio_byte_dtype, compressed_data, {compressed_size}, pressio_data_libc_free_fn, nullptr);
    }

    bool use_sz;
    std::vector<size_t> const& dims;
    int target_level;
    double eb;
  };

  struct decompress_op {
    template <class T>
    pressio_data operator()(unsigned char* data) {
      MGARD::Recomposer<T> recomposer;
      auto data_dec = recomposer.decompress(data, compressed_size, dims);
      return pressio_data::move(pressio_dtype_from_type<T>(), data_dec, dims, pressio_data_libc_free_fn, nullptr);
    }

    size_t compressed_size;
    std::vector<size_t> const& dims;
  };

class mgardx_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:abs", eb);
    set(options, "mgardx:error_bound", eb);
    set(options, "mgardx:target_level", target_level);
    set(options, "mgardx:use_sz", use_sz);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(experimental MGARD implementation for research purposes)");
    set(options, "mgardx:error_bound", "absolute error bound");
    set(options, "mgardx:target_level", "number of target levels to decompose to");
    set(options, "mgardx:use_sz", "use SZ after composing");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "pressio:abs", &eb);
    get(options, "mgardx:error_bound", &eb);
    get(options, "mgardx:target_level", &target_level);
    get(options, "mgardx:use_sz", &use_sz);
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    std::vector<size_t> dimensions{
      input->get_dimension(0),
      input->get_dimension(1),
      input->get_dimension(2),
    };
    std::replace(dimensions.begin(), dimensions.end(), 0, 1);
    if(input->dtype() == pressio_float_dtype) {
      *output = compress_op{static_cast<bool>(use_sz), dimensions, target_level, eb}(static_cast<float*>(input->data()));
    } else if(input->dtype() == pressio_double_dtype) {
      *output = compress_op{static_cast<bool>(use_sz), dimensions, target_level, eb}(static_cast<double*>(input->data()));
    }
    return 0;
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    std::vector<size_t> dimensions{
      output->get_dimension(0),
      output->get_dimension(1),
      output->get_dimension(2),
    };
    std::replace(dimensions.begin(), dimensions.end(), 0, 1);

    if(output->dtype() == pressio_float_dtype) {
      *output = decompress_op{input->get_dimension(0), dimensions}.template operator()<float>(static_cast<unsigned char*>(input->data()));
    } else if(output->dtype() == pressio_double_dtype) {
      *output = decompress_op{input->get_dimension(0), dimensions}.template operator()<double>(static_cast<unsigned char*>(input->data()));
    } else {
      return set_error(1, "unsupported type");
    }
    return 0;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "mgardx"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<mgardx_compressor_plugin>(*this);
  }

  int32_t use_sz = true;
  int32_t target_level = 3;
  double eb = 1e-5;
};

pressio_register plugin(compressor_plugins(), "mgardx", []() {
  return compat::make_unique<mgardx_compressor_plugin>();
});

} } }
