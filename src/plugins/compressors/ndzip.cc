#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include <ndzip/ndzip.hh>

namespace libpressio { namespace ndzip_ns {

  struct dispatcher {

    struct compress_action {
      template <template <typename, unsigned> typename Encoder, class Dtype, size_t Dims>
      pressio_data operator()(pressio_data const& input, std::vector<size_t> const& dims) const {
        ndzip::extent<Dims> extent;
        std::vector<unsigned> ndzip_size(Dims);
        for (size_t i = 0; i < Dims; ++i) {
          extent[i] = dims.at(i);
          ndzip_size[i] = dims.at(i);
        }
        Encoder<Dtype, Dims> encoder;
        ndzip::slice<const Dtype, Dims> in(static_cast<const Dtype*>(input.data()), extent);
        const auto max_compressed_chunk_length = ndzip::compressed_size_bound<Dtype>(extent);
        auto out= std::make_unique<std::byte[]>(max_compressed_chunk_length);
        auto compressed_size = encoder.compress(in, out.get());
        pressio_data ret = pressio_data::copy(pressio_byte_dtype, out.get(), {compressed_size});
        return ret;
      }
    };

    struct decompress_action {
      template <template <typename, unsigned> typename Encoder, class Dtype, size_t Dims>
      pressio_data operator()(pressio_data const& input, std::vector<size_t> const& dims) const {
        ndzip::extent<Dims> extent;
        std::vector<unsigned> ndzip_size(Dims);
        size_t total_size = 1;
        for (size_t i = 0; i < Dims; ++i) {
          extent[i] = dims.at(i);
          ndzip_size[i] = dims.at(i);
          total_size *= dims.at(i);
        }
        Encoder<Dtype, Dims> encoder;
        auto out= std::make_unique<Dtype[]>(total_size);
        ndzip::slice<Dtype, Dims> slice(static_cast<Dtype*>(out.get()), extent);

        auto compressed_size = encoder.decompress(
              input.data(), input.size_in_bytes(), slice
            );
        if(compressed_size == 0) {
          throw std::runtime_error("compression failed");
        }
        pressio_data ret = pressio_data::copy(pressio_dtype_from_type<Dtype>(), out.get(), dims);
        return ret;
      }
    };

    // operator() -> impl_typed -> impl_sized
    pressio_data compress(pressio_data const& input) const {
      return impl(input, compress_action{});
    }

    pressio_data decompress(pressio_data const& input) const {
      return impl(input, decompress_action{});
    }

    template <class Action>
    pressio_data impl(pressio_data const& input, Action const& action) const {
      switch(dtype) {
        case pressio_float_dtype:
          return impl_typed<float>(input, action);
        case pressio_double_dtype:
          return impl_typed<double>(input, action);
        default:
          throw std::runtime_error("unsupported dtype, only float and double are supported");
      }
    }

    template <class Dtype, class Action>
    pressio_data impl_typed(pressio_data const& input, Action const& action) const {
      switch (dims.size()) {
        case 1:
          return impl_sized<Dtype, 1>(input, action);
        case 2:
          return impl_sized<Dtype, 2>(input, action);
        case 3:
          return impl_sized<Dtype, 3>(input, action);
        default:
          throw std::runtime_error("unsupported num_dimensions " + std::to_string(dims.size()));
      }
    }

    template <class Dtype, size_t Dims, class Action>
    pressio_data impl_sized(pressio_data const& input, Action const& action) const {
      if (encoder == "cpu") {
        return action.template operator()<ndzip::cpu_encoder, Dtype, Dims>(input, dims);
#if NDZIP_OPENMP_SUPPORT
      } else if (encoder == "cpu-mt") {
        return action.template operator()<ndzip::mt_cpu_encoder, Dtype, Dims>(input, dims);
#endif
#if NDZIP_HIPSYCL_SUPPORT
      } else if (encoder == "sycl") {
        return action.template operator()<ndzip::sycl_encoder, Dtype, Dims>(data, dims);
#endif
#if NDZIP_CUDA_SUPPORT
      } else if (encoder == "cuda") {
        return action.template operator()<ndzip::cuda_encoder, Dtype, Dims>(input, dims);
#endif
      } else {
        throw std::runtime_error("invalid encoder " + encoder);
      }
    }

    std::string const& encoder;
    std::vector<size_t> const& dims;
    pressio_dtype dtype;
  };

class ndzip_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "ndzip:executor", executor);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    set(options, "ndzip:has_mt_cpu", NDZIP_OPENMP_SUPPORT);
    set(options, "ndzip:has_cuda", NDZIP_CUDA_SUPPORT);
    set(options, "ndzip:has_sycl", NDZIP_HIPSYCL_SUPPORT);
    std::vector<std::string> executor_options;
    executor_options.emplace_back("cpu");
#if NDZIP_OPENMP_SUPPORT
    executor_options.emplace_back("cpu-mt");
#endif
#if NDZIP_CUDA_SUPPORT
    executor_options.emplace_back("cuda");
#endif
#if NDZIP_HIPSYCL_SUPPORT
    executor_options.emplace_back("sycl");
#endif
    set(options, "ndzip:executor", executor_options);
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(A High-Throughput Parallel Lossless Compressor for Scientific Data)");
    set(options, "ndzip:executor", "which executor to use");
    set(options, "ndzip:has_mt_cpu", "compiled with OpenMP support");
    set(options, "ndzip:has_cuda", "compiled with Cuda support");
    set(options, "ndzip:has_sycl", "compiled with Sycl support");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "ndzip:executor", &executor);
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    try {
      std::vector<size_t> lp_dims = input->normalized_dims();
      std::vector<size_t> native_dims(compat::rbegin(lp_dims), compat::rend(lp_dims));

      dispatcher dispatch{executor, native_dims, input->dtype()};
      *output = dispatch.compress(*input);
      return 0;
    } catch(std::exception &ex) {
      return set_error(1, ex.what());
    }
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    try {
      std::vector<size_t> lp_dims = output->normalized_dims();
      std::vector<size_t> native_dims(compat::rbegin(lp_dims), compat::rend(lp_dims));

      dispatcher dispatch{executor, native_dims, output->dtype()};
      *output = dispatch.decompress(*input);
      return 0;
    } catch (std::exception &ex){ 
      return set_error(1, ex.what());
    }
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "ndzip"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<ndzip_compressor_plugin>(*this);
  }

  std::string executor = "cpu";

};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "ndzip", []() {
  return compat::make_unique<ndzip_compressor_plugin>();
});

} }
