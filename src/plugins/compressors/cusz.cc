#define PSZ_USE_CUDA 1

#include <set>
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include <cusz/cusz.h>
#include <cusz/context.h>
#include <cusz/tehm.hh>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusz/cusz_version.h>
#include <cusz/utils/viewer.hh>

namespace libpressio { namespace compressors { namespace cusz_ns {

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
    set(options, "cusz:cuda_stream", "which cudaStream_t to use");
    set(options, "pressio:cuda_stream", "which cudaStream_t to use");
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
  int compress_typed(pressio_data const* real_input, pressio_data* output) {


    pressio_data input = domain_manager().make_readable(domain_plugins().build("cudamalloc"), *real_input);
    auto const dims = input.normalized_dims(4, 1);
    T* d_uncomp = (T*)input.data();

    psz_header header;
    psz::TimeRecord timerecord;
    psz_len3 uncomp_len = psz_len3{dims[0], dims[1], dims[2]};
    psz_compressor* comp = psz_create(to_cuszdtype(input.dtype()), uncomp_len, to_cusz_predictor_type(predictor),
        radius, Huffman); // codectype Huffman is hardcoded. (v0.10rc)
    uint8_t* ptr_compressed;
    size_t compressed_len;
    psz_compress(
        comp, d_uncomp, uncomp_len, err_bnd, to_cuszmode(eb_mode), 
        &ptr_compressed, &compressed_len, &header, &timerecord, *stream.get());

    *output = pressio_data::move(pressio_byte_dtype, ptr_compressed, {compressed_len}, domain_plugins().build("cudamalloc"));
    *output = domain_manager().make_readable(domain_plugins().build("malloc"), std::move(*output));
    memcpy(output->data(), &header, sizeof(psz_header));

    //call when we are done
    psz_release(comp);

    return 0;
  }

  template<class T>
  int decompress_typed(pressio_data const* real_input, pressio_data* output) {
    auto const dims = output->normalized_dims(3, 1);
    *output  = domain_manager().make_writeable(domain_plugins().build("cudamalloc"), std::move(*output));
    T* d_decomp = static_cast<T*>(output->data());

    pressio_data cpu_input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
    psz_header* header = (psz_header*)cpu_input.data();
    auto comp_len = pszheader_filesize(header);
    psz_len3 decomp_len = psz_len3{dims[0], dims[1], dims[2]};  // x, y, z
                                                                //
    pressio_data gpu_input = domain_manager().make_readable(domain_plugins().build("cudamalloc"), *real_input);
    uint8_t* ptr_compressed = (uint8_t*)gpu_input.data();

    psz::TimeRecord timerecord;
    psz_compressor* comp = psz_create_from_header(header);
    psz_decompress(comp, ptr_compressed, comp_len, d_decomp, decomp_len, (void*)&timerecord, *stream.get());

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

pressio_register registration(compressor_plugins(), "cusz", []() {
  return compat::make_unique<cusz_compressor_plugin>();
});

} }
}
