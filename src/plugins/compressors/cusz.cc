#define PSZ_USE_CUDA 1

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

    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "cusz:mode_str", std::vector<std::string>{"abs", "rel"});
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(A GPU based implementation of SZ for Nvidia GPUs)");
    set(options, "cusz:mode_str", "error bound mode");
    set(options, "cusz:bound", "bound of the error bound");
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
    [[deprecated("Only `lorenzo` instead of multiple variants")]]
    if (s == "lorenzo0") return Lorenzo;
    else if (s == "lorenzo") return Lorenzo;
    else if (s == "lorenzoi") return Lorenzo;
    else if (s == "spline") return Spline;
    else throw std::domain_error("unsupported predictor_type: " + s);
  }
  psz_hfpartype to_huffman_style(std::string const& s) {
    if(s == "coarse") return Coarse;
    else if(s == "fine") return Fine;
    else throw std::domain_error("unsupported codec type: " + s);
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
      if (mode == "abs") return Abs;
      else if (mode == "rel") return Rel;
      else throw std::runtime_error("unsupported mode " + mode);
  }
  auto to_bookstyle(std::string const& s) {
      if(s == "") return Canonical;
      else if(s == "") return Sword;
      else if(s == "") return Mword;
      else throw std::runtime_error("unknown bookstyle " + s); 
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
        .predictor = pszpredictor{.type = to_cusz_predictor_type(predictor)},
        .quantizer = pszquantizer{.radius = radius},
        .hfcoder = pszhfrc{
            .book = to_bookstyle(bookstyle),
            .style = to_huffman_style(huffman_coding_style),
            .booklen = booklen,
            .coarse_pardeg = coarse_pardeg 
        },
        .max_outlier_percent = max_outlier_percent};
    pszcompressor* comp = psz_create(work, to_cuszdtype(input->dtype()));
    pszctx* ctx = new pszctx{
        .demodata_name = "",
        .infile = "",
        .original_file = "",
        .opath = "",
        .mode = to_cuszmode(eb_mode),
        .eb = err_bnd
    };
    pszlen uncomp_len = pszlen{{dims[0]}, {dims[1]}, {dims[2]}, {dims[3]}};
    psz::TimeRecord compress_timerecord;
    psz_compress_init(comp, uncomp_len, ctx);
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
    cudaMallocAsync(&d_decomp, output->size_in_bytes(), stream);
    uint8_t* ptr_compressed;
    pszheader header;
    size_t compressed_len = input->size_in_bytes() - sizeof(header);
    if(isDevicePtr(input->data())) {
        cudaMemcpyAsync(&header, input->data(), sizeof(header), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        ptr_compressed = (uint8_t*)input->data()+sizeof(header);
    } else {
        memcpy(&header, input->data(), sizeof(header));
        cudaMallocAsync(&ptr_compressed, compressed_len, stream);
        cudaMemcpyAsync(ptr_compressed, (uint8_t*)input->data()+sizeof(header), compressed_len, cudaMemcpyHostToDevice, stream);
    }

    psz::TimeRecord decompress_timerecord;
    pszlen decomp_len = pszlen{{dims[0]}, {dims[1]}, {dims[2]}, {dims[3]}};  // x, y, z, w
    pszframe* work = new pszframe{
        .predictor = pszpredictor{.type = to_cusz_predictor_type(predictor)},
        .quantizer = pszquantizer{.radius = radius},
        .hfcoder = pszhfrc{
            .book = to_bookstyle(bookstyle),
            .style = to_huffman_style(huffman_coding_style),
            .booklen = booklen,
            .coarse_pardeg = coarse_pardeg 
        },
        .max_outlier_percent = max_outlier_percent};
    pszcompressor* comp = psz_create(work, to_cuszdtype(output->dtype()));
    psz_decompress_init(comp, &header);
    psz_decompress(
        comp, ptr_compressed, compressed_len, d_decomp, decomp_len,
        (void*)&decompress_timerecord, stream);

    if(isDevicePtr(input->data())) {
        if(output->has_data() && isDevicePtr(output->data()) && compressed_len <= output->capacity_in_bytes()) {
            //copy to existing device ptr
            cudaMemcpyAsync(output->data(), d_decomp, output->size_in_bytes(), cudaMemcpyDeviceToDevice, stream);
        } else {
            //copy to new device ptr
            T* buf_uncompressed;
            cudaMallocAsync(&buf_uncompressed, output->size_in_bytes(), stream);
            cudaMemcpyAsync(buf_uncompressed, d_decomp, output->size_in_bytes(), cudaMemcpyDeviceToDevice, stream);
            cudaStreamSynchronize(stream);
            *output = pressio_data::move(output->dtype(),
                    buf_uncompressed, output->dimensions(),
                    [](void* data, void*){ cudaFree(data);}, nullptr
                    );
        }
    } else {
        //copy to host pointer
        if(output->has_data()) {
            cudaMemcpyAsync(output->data(), d_decomp, output->size_in_bytes(), cudaMemcpyDeviceToHost, stream);
        } else {
            *output = pressio_data::owning(output->dtype(), output->dimensions());
            cudaMemcpyAsync(output->data(), d_decomp, output->size_in_bytes(), cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);
        cudaFree(ptr_compressed);
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
  std::string bookstyle = "";
  float max_outlier_percent;
  int32_t radius = 512;
  int32_t booklen = 0;
  int32_t coarse_pardeg  = 0;

};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "cusz", []() {
  return compat::make_unique<cusz_compressor_plugin>();
});

} }

