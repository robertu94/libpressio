#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"

#include <SZ3/compressor/SZBlockInterpolationCompressor.hpp>
#include <SZ3/compressor/SZInterpolationCompressor.hpp>
#include <SZ3/lossless/Lossless_zstd.hpp>
#include <SZ3/def.hpp>

namespace libpressio { namespace sz_interp_ns {

  void delete_free_fn_uchar(void* d, void*) {
    delete[] reinterpret_cast<unsigned char*>(d);
  }

  template <class T> 
  auto  delete_free_fn_typed() {
    return [](void * d, void*) {
      delete[] static_cast<T*>(d);
    };
  }

  struct invoke_sz_interp_compress {
    template <class T>
    pressio_data operator()(T * begin, T *) {
      switch(dims.size()) {
        case 1:
          return invoke_sized<T,1>(begin);
        case 2:
          return invoke_sized<T,2>(begin);
        case 3:
          return invoke_sized<T,3>(begin);
        case 4:
          return invoke_sized<T,4>(begin);
        default:
          throw std::runtime_error("unsupported dimension");
      }
    }
    

    template <class T, size_t N>
    pressio_data invoke_sized(T * begin){
      std::array<size_t,N> dims_arr;
      std::copy(dims.rbegin(), dims.rend(), dims_arr.begin());
      auto sz = SZ::SZInterpolationCompressor<T, N, SZ::LinearQuantizer<T>, SZ::HuffmanEncoder<int32_t>, SZ::Lossless_zstd>(
          SZ::LinearQuantizer<T>(error_bound),
          SZ::HuffmanEncoder<int32_t>(),
          SZ::Lossless_zstd(),
          dims_arr,
          interp_block_size,
          intern_op,
          direction_op
          );
      size_t compresesd_size = 0;
      unsigned char* compressed  = sz.compress(begin, compresesd_size);
      std::vector<size_t> c_dims{compresesd_size};
      return pressio_data::move(
          pressio_byte_dtype,
          compressed,
          c_dims.size(),
          c_dims.data(),
          delete_free_fn_uchar,
          nullptr
          );
    }

    std::vector<size_t> dims;
    double error_bound;
    int32_t interp_block_size;
    int32_t intern_op;
    int32_t direction_op;
  };

  struct invoke_sz_interp_decompress {
    pressio_data operator()(pressio_data const * input, pressio_data* output) {
      switch(output->dtype()) {
        case pressio_float_dtype:
          return invoke_typed<float>(input, output);
        case pressio_double_dtype:
          return invoke_typed<double>(input, output);
        case pressio_int8_dtype:
          return invoke_typed<int8_t>(input, output);
        case pressio_int16_dtype:
          return invoke_typed<int16_t>(input, output);
        case pressio_int32_dtype:
          return invoke_typed<int32_t>(input, output);
        case pressio_int64_dtype:
          return invoke_typed<int64_t>(input, output);
        case pressio_uint8_dtype:
          //fallthough
        case pressio_byte_dtype:
          return invoke_typed<uint8_t>(input, output);
        case pressio_uint16_dtype:
          return invoke_typed<uint16_t>(input, output);
        case pressio_uint32_dtype:
          return invoke_typed<uint32_t>(input, output);
        case pressio_uint64_dtype:
          return invoke_typed<uint64_t>(input, output);
        default:
          throw std::runtime_error("unsupported dtype");
      }
    }

    template <class T>
    pressio_data invoke_typed(pressio_data const * input, pressio_data* output) {
      switch(output->num_dimensions()) {
        case 1:
          return invoke_sized<T,1>(input, output);
        case 2:
          return invoke_sized<T,2>(input, output);
        case 3:
          return invoke_sized<T,3>(input, output);
        case 4:
          return invoke_sized<T,4>(input, output);
        default:
          throw std::runtime_error("unsupported dimension");
      }
    }
    

    template <class T, size_t N>
    pressio_data invoke_sized(pressio_data const* input, pressio_data * output){
      std::array<size_t,N> dims_arr;
      std::vector<size_t> const& dims = output->dimensions();
      std::copy(dims.rbegin(), dims.rend(), dims_arr.begin());
      auto sz = SZ::SZInterpolationCompressor<T, N, SZ::LinearQuantizer<T>, SZ::HuffmanEncoder<int32_t>, SZ::Lossless_zstd>(
          SZ::LinearQuantizer<T>(error_bound),
          SZ::HuffmanEncoder<int32_t>(),
          SZ::Lossless_zstd(),
          dims_arr,
          interp_block_size,
          intern_op,
          direction_op
          );
      size_t compresesd_size = input->get_dimension(0);
      T* decompressed  = sz.decompress(static_cast<unsigned char*>(input->data()), compresesd_size);
      std::vector<size_t> c_dims{compresesd_size};
      return pressio_data::move(
          output->dtype(),
          decompressed,
          output->dimensions(),
          static_cast<void(*)(void*, void*)>(delete_free_fn_typed<T>()),
          nullptr
          );
    }

    double error_bound;
    int32_t interp_block_size;
    int32_t intern_op;
    int32_t direction_op;
  };

class sz_interp_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "sz_interp:error_bound", error_bound);
    set(options, "sz_interp:interp_block_size", interp_block_size);
    set(options, "sz_interp:intern_op", intern_op);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_multiple));
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(A SZ3 based compressor which uses block based interpolation)");
    set(options, "sz_interp:error_bound", "absolute error bound");
    set(options, "sz_interp:interp_block_size", "size of blocks used for interpolation");
    set(options, "sz_interp:intern_op", "interpolation direction, 0=linear");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "sz_interp:error_bound", &error_bound);
    get(options, "sz_interp:interp_block_size", &interp_block_size);
    get(options, "sz_interp:intern_op", &intern_op);
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    try {
      *output = pressio_data_for_each<pressio_data>(*input, invoke_sz_interp_compress{
            input->dimensions(),
            error_bound,
            interp_block_size,
            intern_op,
            direction_op
          });
    } catch(std::exception const& ex) {
      return set_error(1, ex.what());
    }
    return 0;
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    try {
      invoke_sz_interp_decompress{ error_bound, interp_block_size, intern_op, direction_op }(input,output);
    } catch(std::exception const& ex) {
      return set_error(1, ex.what());
    }
    return 0;
  }

  int major_version() const override { return SZ_VERSION_MAJOR; }
  int minor_version() const override { return SZ_VERSION_MINOR; }
  int patch_version() const override { return SZ_VERSION_RELEASE; }
  const char* version() const override { return SZ_VERSION_STRING; }
  const char* prefix() const override { return "sz_interp"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<sz_interp_compressor_plugin>(*this);
  }

  double error_bound = 1e-6;
  int32_t interp_block_size = 32;
  int32_t intern_op = 0;
  int32_t direction_op = 0;

};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "sz_interp", []() {
  return compat::make_unique<sz_interp_compressor_plugin>();
});

} }
