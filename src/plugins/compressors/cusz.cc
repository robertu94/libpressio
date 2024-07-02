#define PSZ_USE_CUDA 1

#include <set>
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include <cusz/cusz.h>
#include <cusz/context.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusz/cusz_version.h>
#include <cusz/utils/viewer.hh>

namespace libpressio { namespace cusz_ns {

    const std::map<std::string, decltype(Rel)> bounds {
        {"abs", Abs},
            {"rel", Rel},
    };
    const std::map<std::string, decltype(NVGPU)> devices {
        {"cuda",NVGPU},
            {"amd",AMDGPU},
            {"intel",INTELGPU},
            {"cpu",CPU},
    };
    const std::map<std::string, decltype(Abs)> cuszmode {
        {"abs", Abs},
            {"rel", Rel},
    };
    const std::map<std::string, decltype(Lorenzo)> predictors {
        {"lorenzo", Lorenzo},
            {"spline", Spline},
    };

    struct stream_helper {
        stream_helper() {
            cudaStreamCreate(&stream);
        }
        ~stream_helper() {
            if(cleanup) {
                cudaStreamDestroy(stream);
            }
        }
        stream_helper(stream_helper&) {
            cudaStreamCreate(&stream);
        }
        stream_helper& operator=(stream_helper& rhs) {
            if(this == &rhs) return *this;
            cudaStreamCreate(&stream);
            return *this;
        }
        stream_helper& operator=(stream_helper&& rhs) {
            if(this == &rhs) return *this;
            rhs.cleanup=false;
            stream = rhs.stream;
            return *this;
        }
        void reset(cudaStream_t& new_stream) {
            cudaStreamDestroy(stream);
            stream = new_stream;
        }
        stream_helper(stream_helper&& rhs): stream(rhs.stream) {
            rhs.cleanup=false;
        }
        cudaStream_t& operator* (){
            return stream;
        }
        cudaStream_t* get() const {
            return const_cast<cudaStream_t*>(&stream);
        }
        cudaStream_t stream;
        bool cleanup = true;
    };

template <class T>
std::vector<std::string> to_keys(std::map<std::string, T> const& map) {
    std::set<std::string> s;
    for (auto const& i : map) {
        s.emplace(i.first);
    }
    return std::vector<std::string>(s.begin(), s.end());
}

class cusz_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    if(eb_mode == "abs") {
      set(options, "pressio:abs", err_bnd);
    } else {
      set_type(options, "pressio:abs", pressio_option_double_type);
    }

    if(eb_mode == "rel") {
      set(options, "pressio:rel", err_bnd);
    } else {
      set_type(options, "pressio:rel", pressio_option_double_type);
    }

    set(options, "cusz:mode_str", eb_mode);
    set(options, "cusz:bound", err_bnd);
    set(options, "cusz:coarse_pardeg", coarse_pardeg);
    set(options, "cusz:booklen", booklen);
    set(options, "cusz:radius", radius);
    set(options, "cusz:max_outlier_percent", max_outlier_percent);
    set(options, "cusz:device", device);
    set(options, "cusz:predictor", predictor);
    set(options, "pressio:cuda_stream", static_cast<void*>(stream.get()));
    set(options, "cusz:cuda_stream", static_cast<void*>(stream.get()));

    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "cusz:mode_str", to_keys(bounds));
    set(options, "cusz:device", to_keys(devices));
    set(options, "cusz:predictor", to_keys(predictors));
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    
    std::vector<pressio_configurable const*> invalidation_children {}; 
    set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{"pressio:abs", "pressio:rel"}));
    std::vector<std::string> error_invalidations {"cusz:mode_str", "cusz:bound",  "cusz:radius", "cusz:max_outlier_percent", "cusz:predictor", "pressio:abs", "pressio:rel"}; 
    std::vector<std::string> invalidations {"cusz:mode_str", "cusz:bound", "cusz:coarse_pardeg", "cusz:booklen", "cusz:radius", "cusz:max_outlier_percent", "cusz:predictor", "pressio:abs", "pressio:rel"}; 
    std::vector<std::string> runtime_invalidations {"cusz:mode_str", "cusz:bound", "cusz:coarse_pardeg", "cusz:booklen", "cusz:radius", "cusz:max_outlier_percent", "cusz:device", "cusz:predictor", "pressio:abs", "pressio:rel"}; 
    
    set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, error_invalidations));
    set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
    set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, runtime_invalidations));

    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(A GPU based implementation of SZ for Nvidia GPUs)");
    set(options, "cusz:mode_str", "error bound mode");
    set(options, "cusz:bound", "bound of the error bound");
    set(options, "cusz:coarse_pardeg", "paralellism degree for the huffman encoding stage");
    set(options, "cusz:booklen", "huffman encoding booklength");
    set(options, "cusz:radius", "quantizer radius");
    set(options, "cusz:max_outlier_percent", "max outlier percent");
    set(options, "cusz:device", "execucution device");
    set(options, "cusz:predictor", "predictor style");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    if(get(options, "pressio:abs", &err_bnd) == pressio_options_key_set) {
      eb_mode = "abs";
    }
    if(get(options, "pressio:rel", &err_bnd) == pressio_options_key_set) {
      eb_mode = "rel";
    }
    get(options, "cusz:mode_str", &eb_mode);
    get(options, "cusz:bound", &err_bnd);
    get(options, "cusz:coarse_pardeg", &coarse_pardeg);
    get(options, "cusz:booklen", &booklen);
    get(options, "cusz:radius", &radius);
    get(options, "cusz:max_outlier_percent", &max_outlier_percent);
    get(options, "cusz:device", &device);
    get(options, "cusz:predictor", &predictor);
  
    // arbitrary stream
    void* void_stream;
    if(get(options, "pressio:cuda_stream", &void_stream) == pressio_options_key_set) {
        cudaStream_t* cuda_stream = static_cast<cudaStream_t*>(void_stream);
        stream.reset(*cuda_stream);
    }
    if(get(options, "cusz:cuda_stream", &void_stream) == pressio_options_key_set) {
        cudaStream_t* cuda_stream = static_cast<cudaStream_t*>(void_stream);
        stream.reset(*cuda_stream);
    }

    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override {
    try{
      switch(input->dtype()) {
        case pressio_float_dtype:
          return compress_typed<float>(input, output);
        case pressio_double_dtype:
          return compress_typed<double>(input, output);
        default:
          return set_error(1, "unsupported dtype");
      }
    } catch (std::exception const&ex) {
      return set_error(2, ex.what());
    }
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    try{
      switch(output->dtype()) {
        case pressio_float_dtype:
          return decompress_typed<float>(input, output);
        case pressio_double_dtype:
          return decompress_typed<double>(input, output);
        default:
          return set_error(1, "unsupported dtype");
      }
    } catch (std::exception const&ex) {
      return set_error(2, ex.what());
    }
  }

  psz_predtype to_cusz_predictor_type(std::string const& s) {
      try {
        return predictors.at(s);
      } catch(std::out_of_range const& ex) {
        throw std::domain_error("unsupported predictor_type: " + s);
      }
  }

  int cuda_error(cudaError_t ec) {
    return set_error(12, cudaGetErrorString(ec));
  }

  int cusz_error(pszerror ec) {
    switch(ec) {
      case CUSZ_FAIL_ONDISK_FILE_ERROR:
        return set_error(3, "ondisk file error");
      case CUSZ_FAIL_DATA_NOT_READY:
        return set_error(4, "data not ready");
      case CUSZ_FAIL_GPU_MALLOC:
        return set_error(5, "gpu malloc");
      case CUSZ_FAIL_GPU_MEMCPY:
        return set_error(6, "gpu memcpy");
      case CUSZ_FAIL_GPU_ILLEGAL_ACCESS:
        return set_error(7, "gpu illegal access");
      case CUSZ_FAIL_GPU_OUT_OF_MEMORY:
        return set_error(8, "gpu out of memory");
      case CUSZ_FAIL_INCOMPRESSIABLE:
        return set_error(9, "incompressible");
      default:
        return set_error(10, "unknown error");
    }
  }

#define lp_check_cuda_error(call) { \
    cudaError ec = (call); \
    if(ec != cudaSuccess) \
      return cuda_error(ec);\
    } \
   \

  psz_dtype to_cuszdtype(pressio_dtype t) {
      switch(t) {
          case pressio_float_dtype:
              return F4;
          case pressio_double_dtype:
              return F8;
          case pressio_int8_dtype:
              return I1;
          case pressio_int16_dtype:
              return I2;
          case pressio_int32_dtype:
              return I4;
          case pressio_int64_dtype:
              return I8;
          case pressio_uint8_dtype:
              return U1;
          case pressio_uint16_dtype:
              return U2;
          case pressio_uint32_dtype:
              return U4;
          case pressio_uint64_dtype:
              return U8;
          default:
              throw std::runtime_error("failed to convert type");
      }
  }

  auto to_cuszmode(std::string const& mode) {
      try {
        return cuszmode.at(mode);
      } catch(std::out_of_range const& ex) {
        throw std::runtime_error("unsupported mode " + mode);
      }
  }
  auto to_device(std::string const& s) {
      try {
        return devices.at(s);
      } catch(std::out_of_range const& ex) {
        throw std::runtime_error("unsupported device " + s);
      }
  }
  bool isDevicePtr(void* ptr) const {
    cudaPointerAttributes attrs;
    cudaPointerGetAttributes(&attrs, ptr);
    return (attrs.type == cudaMemoryTypeDevice);
  }

  template<class T>
  int compress_typed(pressio_data const* input, pressio_data* output) {

    auto const dims = input->normalized_dims(4, 1);

    uint8_t* compressed_buf;
    uint8_t* ptr_compressed;
    size_t compressed_len;
    T* d_uncomp;
    if(isDevicePtr(input->data())) {
        d_uncomp = (T*)input->data();
    } else {
        lp_check_cuda_error(cudaMallocAsync(&d_uncomp, input->size_in_bytes(), *stream.get()));
        lp_check_cuda_error(cudaMemcpyAsync(d_uncomp, input->data(), input->size_in_bytes(), cudaMemcpyHostToDevice, *stream.get()));
    }

    psz_header header;
    void* compress_timerecord;
    psz_len3 uncomp_len = psz_len3{{dims[0]}, {dims[1]}, {dims[2]}};
    psz_compressor* comp = psz_create(to_cuszdtype(input->dtype()), uncomp_len, to_cusz_predictor_type(predictor),
        radius, Huffman); // codectype Huffman is hardcoded. (v0.10rc)
    psz_compress(
        comp, d_uncomp, uncomp_len, err_bnd, to_cuszmode(eb_mode), 
        &ptr_compressed, &compressed_len, &header, compress_timerecord, *stream.get());


    if(isDevicePtr(input->data())) {
        if(output->has_data() && isDevicePtr(output->data()) && compressed_len <= output->capacity_in_bytes()) {
            //copy off the compressed data
            //compressed data needs to be copied before the compressor is destructed
            lp_check_cuda_error(cudaMemcpyAsync(
                    (uint8_t*)output->data(), &header, sizeof(header),
                    cudaMemcpyDeviceToDevice, *stream.get()));
            lp_check_cuda_error(cudaMemcpyAsync(
                    (uint8_t*)output->data()+sizeof(header), ptr_compressed, compressed_len,
                    cudaMemcpyDeviceToDevice, *stream.get()));
            output->set_dimensions({compressed_len + sizeof(header)});
            output->set_dtype(pressio_byte_dtype);
            lp_check_cuda_error(cudaStreamSynchronize(*stream.get()));

        } else {
            lp_check_cuda_error(cudaMallocAsync(&compressed_buf, compressed_len+sizeof(header), *stream.get()));
            lp_check_cuda_error(cudaMemcpyAsync(
                    (uint8_t*)output->data(), &header, sizeof(header),
                    cudaMemcpyDeviceToDevice, *stream.get()));
            lp_check_cuda_error(cudaMemcpyAsync(
                  compressed_buf+sizeof(header), ptr_compressed, compressed_len,
                  cudaMemcpyDeviceToDevice, *stream.get()));
            lp_check_cuda_error(cudaStreamSynchronize(*stream.get()));
            *output = pressio_data::move(
                    pressio_byte_dtype, compressed_buf,
                    {compressed_len}, [](void* data, void*){ cudaFree(data);}, nullptr
                    );
        }
        lp_check_cuda_error(cudaFree(d_uncomp));
    } else {
        //return data to host
        if(output->has_data()) {
            memcpy(output->data(), &header, sizeof(header));
            lp_check_cuda_error(cudaMemcpyAsync(
                  (uint8_t*)output->data() + sizeof(header), ptr_compressed, compressed_len,
                  cudaMemcpyDeviceToHost, *stream.get()));
            output->set_dimensions({compressed_len+sizeof(header)});
            output->set_dtype(pressio_byte_dtype);
            lp_check_cuda_error(cudaStreamSynchronize(*stream.get()));
        } else {
            *output = pressio_data::owning(pressio_byte_dtype, {sizeof(header)+compressed_len});
            memcpy(output->data(), &header, sizeof(header));
            lp_check_cuda_error(cudaMemcpyAsync(
                  (uint8_t*)output->data()+sizeof(header), ptr_compressed, compressed_len,
                  cudaMemcpyDeviceToHost, *stream.get()));
            lp_check_cuda_error(cudaStreamSynchronize(*stream.get()));
        }
    }

    //call when we are done
    psz_release(comp);
    // cudaStreamDestroy(stream);

    return 0;
  }

  template<class T>
  int decompress_typed(pressio_data const* input, pressio_data* output) {
    auto const dims = output->normalized_dims(4, 1);
    T *d_decomp;
    lp_check_cuda_error(cudaMallocAsync(&d_decomp, output->size_in_bytes(), *stream.get()));
    uint8_t* ptr_compressed;
    psz_header header;
    size_t compressed_len = input->size_in_bytes() - sizeof(header);
    if(isDevicePtr(input->data())) {
        lp_check_cuda_error(cudaMemcpyAsync(&header, input->data(), sizeof(header), cudaMemcpyDeviceToHost, *stream.get()));
        lp_check_cuda_error(cudaStreamSynchronize(*stream.get()));
        ptr_compressed = (uint8_t*)input->data()+sizeof(header);
    } else {
        memcpy(&header, input->data(), sizeof(header));
        lp_check_cuda_error(cudaMallocAsync(&ptr_compressed, compressed_len, *stream.get()));
        lp_check_cuda_error(cudaMemcpyAsync(ptr_compressed, (uint8_t*)input->data()+sizeof(header), compressed_len, cudaMemcpyHostToDevice, *stream.get()));
    }

    void* decompress_timerecord;
    psz_len3 decomp_len = psz_len3{{dims[0]}, {dims[1]}, {dims[2]}};  // x, y, z
    psz_compressor* comp = psz_create_from_header(&header);
    psz_decompress(comp, ptr_compressed, compressed_len, d_decomp, decomp_len, decompress_timerecord, *stream.get());

    if(isDevicePtr(input->data())) {
        if(output->has_data() && isDevicePtr(output->data()) && compressed_len <= output->capacity_in_bytes()) {
            //copy to existing device ptr
            lp_check_cuda_error(cudaMemcpyAsync(output->data(), d_decomp, output->size_in_bytes(), cudaMemcpyDeviceToDevice, *stream.get()));
        } else {
            //copy to new device ptr
            T* buf_uncompressed;
            lp_check_cuda_error(cudaMallocAsync(&buf_uncompressed, output->size_in_bytes(), *stream.get()));
            lp_check_cuda_error(cudaMemcpyAsync(buf_uncompressed, d_decomp, output->size_in_bytes(), cudaMemcpyDeviceToDevice, *stream.get()));
            lp_check_cuda_error(cudaStreamSynchronize(*stream.get()));
            *output = pressio_data::move(output->dtype(),
                    buf_uncompressed, output->dimensions(),
                    [](void* data, void*){ cudaFree(data);}, nullptr
                    );
        }
    } else {
        //copy to host pointer
        if(output->has_data()) {
            lp_check_cuda_error(cudaMemcpyAsync(output->data(), d_decomp, output->size_in_bytes(), cudaMemcpyDeviceToHost, *stream.get()));
        } else {
            *output = pressio_data::owning(output->dtype(), output->dimensions());
            lp_check_cuda_error(cudaMemcpyAsync(output->data(), d_decomp, output->size_in_bytes(), cudaMemcpyDeviceToHost, *stream.get()));
        }
        lp_check_cuda_error(cudaStreamSynchronize(*stream.get()));
        lp_check_cuda_error(cudaFree(ptr_compressed));
    }
    psz_release(comp);
    return 0;
  }


  int major_version() const override { return CUSZ_MAJOR_VERSION; }
  int minor_version() const override { return CUSZ_MINOR_VERSION; }
  int patch_version() const override { return CUSZ_PATCH_VERSION; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "cusz"; }

  pressio_options get_metrics_results_impl() const override {
    pressio_options metrics;
    return metrics;
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<cusz_compressor_plugin>(*this);
  }

  stream_helper stream; 

  double err_bnd = 1e-5;
  std::string eb_mode = "abs";
  std::string predictor = "lorenzo";
  std::string device = "cuda";
  float max_outlier_percent = 10.0;
  int32_t radius = 512;
  int32_t booklen = 0;
  int32_t coarse_pardeg  = 0;

};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "cusz", []() {
  return compat::make_unique<cusz_compressor_plugin>();
});

} }

