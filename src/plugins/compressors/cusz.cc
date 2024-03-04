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
        {"rel",Rel},
            {"rel",Abs},
    };
    const std::map<std::string, decltype(NVGPU)> devices {
        {"cuda",NVGPU},
            {"amd",AMDGPU},
            {"intel",INTELGPU},
            {"cpu",CPU},
    };
    const std::map<std::string, decltype(Canonical)> bookstyles {
        { "canonical", Canonical},
            { "sword", Sword},
            { "mword", Mword}
    };
    const std::map<std::string, decltype(Abs)> cuszmode {
        {"abs", Abs},
            {"rel", Rel},
    };
    const std::map<std::string, decltype(Lorenzo)> predictors {
        {"lorenzo", Lorenzo},
            {"lorenzoi", Lorenzo},
            {"lorenzo0", Lorenzo},
            {"spline", Spline},
    };
    const std::map<std::string, decltype(Fine)> huffman_styles {
        {"coarse",Coarse},
            {"fine",Fine},
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
    set(options, "cusz:bookstyle", bookstyle);
    set(options, "cusz:radius", radius);
    set(options, "cusz:max_outlier_percent", max_outlier_percent);
    set(options, "cusz:device", device);
    set(options, "cusz:huffman_coding_style", huffman_coding_style);
    set(options, "cusz:predictor", predictor);

    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "cusz:mode_str", to_keys(bounds));
    set(options, "cusz:device", to_keys(devices));
    set(options, "cusz:huffman_coding_style", to_keys(huffman_styles));
    set(options, "cusz:predictor", to_keys(predictors));
    set(options, "cusz:bookstyle", to_keys(bookstyles));
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    
    std::vector<pressio_configurable const*> invalidation_children {}; 
    set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{"pressio:abs", "pressio:rel"}));
    std::vector<std::string> error_invalidations {"cusz:mode_str", "cusz:bound",  "cusz:radius", "cusz:max_outlier_percent", "cusz:huffman_coding_style", "cusz:predictor", "pressio:abs", "pressio:rel"}; 
    std::vector<std::string> invalidations {"cusz:mode_str", "cusz:bound", "cusz:coarse_pardeg", "cusz:booklen", "cusz:bookstyle", "cusz:radius", "cusz:max_outlier_percent", "cusz:huffman_coding_style", "cusz:predictor", "pressio:abs", "pressio:rel"}; 
    std::vector<std::string> runtime_invalidations {"cusz:mode_str", "cusz:bound", "cusz:coarse_pardeg", "cusz:booklen", "cusz:bookstyle", "cusz:radius", "cusz:max_outlier_percent", "cusz:device", "cusz:huffman_coding_style", "cusz:predictor", "pressio:abs", "pressio:rel"}; 
    
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
    set(options, "cusz:bookstyle", "huffman encoding bookstyle");
    set(options, "cusz:radius", "quantizer radius");
    set(options, "cusz:max_outlier_percent", "max outlier percent");
    set(options, "cusz:device", "execucution device");
    set(options, "cusz:huffman_coding_style", "huffman coding style");
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
    get(options, "cusz:bookstyle", &bookstyle);
    get(options, "cusz:radius", &radius);
    get(options, "cusz:max_outlier_percent", &max_outlier_percent);
    get(options, "cusz:device", &device);
    get(options, "cusz:huffman_coding_style", &huffman_coding_style);
    get(options, "cusz:predictor", &predictor);

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
  psz_hfpartype to_huffman_style(std::string const& s) {
      try {
        return huffman_styles.at(s);
      } catch(std::out_of_range const& ex) {
        throw std::domain_error("unsupported huffman style: " + s);
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
  auto to_bookstyle(std::string const& s) {
      try {
        return bookstyles.at(s);
      } catch(std::out_of_range const& ex) {
        throw std::runtime_error("unsupported bookstyle " + s);
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
    cudaStream_t stream;
    lp_check_cuda_error(cudaStreamCreate(&stream));

    auto const dims = input->normalized_dims(4, 1);

    uint8_t* compressed_buf;
    uint8_t* ptr_compressed;
    size_t compressed_len;
    T* d_uncomp;
    if(isDevicePtr(input->data())) {
        d_uncomp = (T*)input->data();
    } else {
        lp_check_cuda_error(cudaMallocAsync(&d_uncomp, input->size_in_bytes(), stream));
        lp_check_cuda_error(cudaMemcpyAsync(d_uncomp, input->data(), input->size_in_bytes(), cudaMemcpyHostToDevice, stream));
    }

    pszheader header;
    pszframe* work = new pszframe{
        pszpredictor{to_cusz_predictor_type(predictor)},
        pszquantizer{radius},
        pszhfrc{
            to_bookstyle(bookstyle),
            to_huffman_style(huffman_coding_style),
            booklen,
            coarse_pardeg 
        },
        max_outlier_percent};
    pszcompressor* comp = psz_create(work, to_cuszdtype(input->dtype()));
    auto ctx = std::make_unique<pszctx>(pszctx{});
    ctx->device = to_device(device);
    ctx->pred_type = to_cusz_predictor_type(predictor);
    // ctx->dbgstr_pred = "";
    // ctx->demodata_name = "";
    // ctx->infile = "";
    // ctx->original_file = "";
    // ctx->opath = "";
    ctx->mode = to_cuszmode(eb_mode);
    ctx->eb = err_bnd;

    pszlen uncomp_len = pszlen{{dims[0]}, {dims[1]}, {dims[2]}, {dims[3]}};
    psz::TimeRecord compress_timerecord;
    psz_compress_init(comp, uncomp_len, ctx.get());
    psz_compress(
        comp, d_uncomp, uncomp_len, &ptr_compressed, &compressed_len, &header,
        (void*)&compress_timerecord, stream);

    if(isDevicePtr(input->data())) {
        if(output->has_data() && isDevicePtr(output->data()) && compressed_len <= output->capacity_in_bytes()) {
            //copy off the compressed data
            //compressed data needs to be copied before the compressor is destructed
            lp_check_cuda_error(cudaMemcpyAsync(
                    (uint8_t*)output->data(), &header, sizeof(header),
                    cudaMemcpyDeviceToDevice, stream));
            lp_check_cuda_error(cudaMemcpyAsync(
                    (uint8_t*)output->data()+sizeof(header), ptr_compressed, compressed_len,
                    cudaMemcpyDeviceToDevice, stream));
            output->set_dimensions({compressed_len + sizeof(header)});
            output->set_dtype(pressio_byte_dtype);
            lp_check_cuda_error(cudaStreamSynchronize(stream));

        } else {
            lp_check_cuda_error(cudaMallocAsync(&compressed_buf, compressed_len+sizeof(header), stream));
            lp_check_cuda_error(cudaMemcpyAsync(
                    (uint8_t*)output->data(), &header, sizeof(header),
                    cudaMemcpyDeviceToDevice, stream));
            lp_check_cuda_error(cudaMemcpyAsync(
                  compressed_buf+sizeof(header), ptr_compressed, compressed_len,
                  cudaMemcpyDeviceToDevice, stream));
            lp_check_cuda_error(cudaStreamSynchronize(stream));
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
                  cudaMemcpyDeviceToHost, stream));
            output->set_dimensions({compressed_len+sizeof(header)});
            output->set_dtype(pressio_byte_dtype);
            lp_check_cuda_error(cudaStreamSynchronize(stream));
        } else {
            *output = pressio_data::owning(pressio_byte_dtype, {sizeof(header)+compressed_len});
            memcpy(output->data(), &header, sizeof(header));
            lp_check_cuda_error(cudaMemcpyAsync(
                  (uint8_t*)output->data()+sizeof(header), ptr_compressed, compressed_len,
                  cudaMemcpyDeviceToHost, stream));
            lp_check_cuda_error(cudaStreamSynchronize(stream));
        }
    }


    //call when we are done
    psz_release(comp);
    cudaStreamDestroy(stream);

    return 0;
  }

  template<class T>
  int decompress_typed(pressio_data const* input, pressio_data* output) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto const dims = output->normalized_dims(4, 1);
    T *d_decomp;
    lp_check_cuda_error(cudaMallocAsync(&d_decomp, output->size_in_bytes(), stream));
    uint8_t* ptr_compressed;
    pszheader header;
    size_t compressed_len = input->size_in_bytes() - sizeof(header);
    if(isDevicePtr(input->data())) {
        lp_check_cuda_error(cudaMemcpyAsync(&header, input->data(), sizeof(header), cudaMemcpyDeviceToHost, stream));
        lp_check_cuda_error(cudaStreamSynchronize(stream));
        ptr_compressed = (uint8_t*)input->data()+sizeof(header);
    } else {
        memcpy(&header, input->data(), sizeof(header));
        lp_check_cuda_error(cudaMallocAsync(&ptr_compressed, compressed_len, stream));
        lp_check_cuda_error(cudaMemcpyAsync(ptr_compressed, (uint8_t*)input->data()+sizeof(header), compressed_len, cudaMemcpyHostToDevice, stream));
    }

    psz::TimeRecord decompress_timerecord;
    pszlen decomp_len = pszlen{{dims[0]}, {dims[1]}, {dims[2]}, {dims[3]}};  // x, y, z, w
    auto work = std::make_unique<pszframe>(pszframe{
        .predictor = pszpredictor{.type = to_cusz_predictor_type(predictor)},
        .quantizer = pszquantizer{.radius = radius},
        .hfcoder = pszhfrc{
            .book = to_bookstyle(bookstyle),
            .style = to_huffman_style(huffman_coding_style),
            .booklen = booklen,
            .coarse_pardeg = coarse_pardeg 
        },
        .max_outlier_percent = max_outlier_percent});
    pszcompressor* comp = psz_create(work.get(), to_cuszdtype(output->dtype()));
    psz_decompress_init(comp, &header);
    psz_decompress(
        comp, ptr_compressed, compressed_len, d_decomp, decomp_len,
        (void*)&decompress_timerecord, stream);

    if(isDevicePtr(input->data())) {
        if(output->has_data() && isDevicePtr(output->data()) && compressed_len <= output->capacity_in_bytes()) {
            //copy to existing device ptr
            lp_check_cuda_error(cudaMemcpyAsync(output->data(), d_decomp, output->size_in_bytes(), cudaMemcpyDeviceToDevice, stream));
        } else {
            //copy to new device ptr
            T* buf_uncompressed;
            lp_check_cuda_error(cudaMallocAsync(&buf_uncompressed, output->size_in_bytes(), stream));
            lp_check_cuda_error(cudaMemcpyAsync(buf_uncompressed, d_decomp, output->size_in_bytes(), cudaMemcpyDeviceToDevice, stream));
            lp_check_cuda_error(cudaStreamSynchronize(stream));
            *output = pressio_data::move(output->dtype(),
                    buf_uncompressed, output->dimensions(),
                    [](void* data, void*){ cudaFree(data);}, nullptr
                    );
        }
    } else {
        //copy to host pointer
        if(output->has_data()) {
            lp_check_cuda_error(cudaMemcpyAsync(output->data(), d_decomp, output->size_in_bytes(), cudaMemcpyDeviceToHost, stream));
        } else {
            *output = pressio_data::owning(output->dtype(), output->dimensions());
            lp_check_cuda_error(cudaMemcpyAsync(output->data(), d_decomp, output->size_in_bytes(), cudaMemcpyDeviceToHost, stream));
        }
        lp_check_cuda_error(cudaStreamSynchronize(stream));
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



  double err_bnd = 1e-5;
  std::string eb_mode = "abs";
  std::string predictor = "lorenzo";
  std::string huffman_coding_style = "coarse";
  std::string bookstyle = "canonical";
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

