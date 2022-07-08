#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include <cusz/cusz.h>
#include <cusz/cuszapi.hh>
#include <cusz/cusz_version.h>

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
    set(options, "cusz:pipeline_type", pipeline_type);
    set(options, "cusz:cusz_len_factor", cusz_len_factor);
    set(options, "cusz:predictor_anchor", predictor_anchor);
    set(options, "cusz:predictor_nondestructive", predictor_nondestructive);
    set(options, "cusz:predictor_type", predictor_type);
    set(options, "cusz:quantization_radius", quantization_radius);
    set(options, "cusz:quantization_delayed", quantization_delayed);
    set(options, "cusz:codec_type", codec_type);
    set(options, "cusz:codec_variable_length", codec_variable_length);
    set(options, "cusz:codec_presumed_density", codec_presumed_density);
    set(options, "cusz:huffman_codec_booktype", huffman_codec_booktype);
    set(options, "cusz:huffman_execution_type", huffman_execution_type);
    set(options, "cusz:huffman_coding", huffman_coding);
    set(options, "cusz:huffman_booklen", huffman_booklen);
    set(options, "cusz:huffman_coarse_pardeg", huffman_coarse_pardeg);

    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "cusz:mode_str", std::vector<std::string>{"abs", "r2r"});
    set(options, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_multiple));
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(A GPU based implementation of SZ for Nvidia GPUs)");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    if(get(options, "pressio:abs", &err_bnd) == pressio_options_key_set) {
      eb_mode = "abs";
    }
    if(get(options, "pressio:rel", &err_bnd) == pressio_options_key_set) {
      eb_mode = "r2r";
    }
    get(options, "cusz:mode_str", &eb_mode);
    get(options, "cusz:bound", &err_bnd);
    get(options, "cusz:pipeline_type", &pipeline_type);
    get(options, "cusz:cusz_len_factor", &cusz_len_factor);
    get(options, "cusz:predictor_anchor", &predictor_anchor);
    get(options, "cusz:predictor_nondestructive", &predictor_nondestructive);
    get(options, "cusz:predictor_type", &predictor_type);
    get(options, "cusz:quantization_radius", &quantization_radius);
    get(options, "cusz:quantization_delayed", &quantization_delayed);
    get(options, "cusz:codec_type", &codec_type);
    get(options, "cusz:codec_variable_length", &codec_variable_length);
    get(options, "cusz:codec_presumed_density", &codec_presumed_density);
    get(options, "cusz:huffman_codec_booktype", &huffman_codec_booktype);
    get(options, "cusz:huffman_execution_type", &huffman_execution_type);
    get(options, "cusz:huffman_coding", &huffman_coding);
    get(options, "cusz:huffman_booklen", &huffman_booklen);
    get(options, "cusz:huffman_coarse_pardeg", &huffman_coarse_pardeg);
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

  cusz_datatype to_cusz_datatype(pressio_dtype dtype) {
    switch(dtype) {
      case pressio_float_dtype: return FP32;
      case pressio_double_dtype: return FP64;
      default: {
        std::stringstream ss;
        ss <<"only float and double supported: "  << dtype;
        throw std::domain_error(ss.str());
       }
    }
  }

  cusz_pipelinetype to_cusz_pipelinetype(std::string const& s) {
    if(s == "auto") return Auto;
    else if(s == "dense") return Dense;
    else if(s == "sparse") return Sparse;
    else throw std::domain_error("unsupported pipeline type: " + s);
  }

  cusz_mode to_cusz_mode(std::string const& s){
    if (s == "abs") return Abs;
    else if (s == "rel") return Rel;
    else throw std::domain_error("unsupported mode: " + s);
  }
  cusz_predictortype to_cusz_predictor_type(std::string const& s) {
    if (s == "lorenzo0") return Lorenzo0;
    else if (s == "lorenzo1" || s == "lorenzoi") return LorenzoI;
    else if (s == "lorenzo2" || s == "lorenzoii") return LorenzoII;
    else if (s == "spline3") return Spline3;
    else throw std::domain_error("unsupported predictor_type: " + s);
  }
  cusz_codectype to_cusz_codec_type(std::string const& s) {
    if(s == "huffman") return Huffman;
    else if(s == "runlength") return RunLength;
    else if(s == "nvcompcascade") return NvcompCascade;
    else if(s == "nvcomplz4") return NvcompLz4;
    else if(s == "nvcompsnappy") return NvcompSnappy;
    else throw std::domain_error("unsupported codec type: " + s);
  }
  cusz_huffman_booktype to_cusz_booktype(std::string const& s) {
    if(s == "tree") return Tree;
    else if(s == "canonical") return Canonical;
    else throw std::domain_error("unsupported book type: " + s);
  }
  cusz_executiontype to_cusz_execution_type(std::string const& s) {
    if(s == "device") return Device;
    if(s == "host") return Host;
    if(s == "none") return None;
    else throw std::domain_error("unsupported execution type: " + s);
  }

  cusz_huffman_codingtype to_cusz_coding_type(std::string const& s) {
    if(s == "coarse") return Coarse;
    if(s == "fine") return Fine;
    else throw std::domain_error("unsupported coding type: " + s);
  }

  int cuda_error(cudaError_t ec) {
    return set_error(12, cudaGetErrorString(ec));
  }

  int cusz_error(cusz_error_status ec) {
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

  template<class T>
  int compress_typed(pressio_data const* input, pressio_data* output) {
    //allocate input device buffer
    T* d_uncompressed;
    size_t uncompressed_alloc_len = std::ceil(static_cast<float>(input->num_elements()) * cusz_len_factor);
    lp_check_cuda_error(cudaMalloc(&d_uncompressed, sizeof(T)* uncompressed_alloc_len));
    lp_check_cuda_error(cudaMemcpy(d_uncompressed, input->data(), input->size_in_bytes(), cudaMemcpyHostToDevice));

    //allocate output host buffer
    cusz_header header;
    uint8_t*    exposed_compressed;

    cudaStream_t stream;
    lp_check_cuda_error(cudaStreamCreate(&stream));
    size_t      compressed_len;
    cusz_framework* framework = new cusz_custom_framework{
        /*datatype*/to_cusz_datatype(input->dtype()),
        /*pipeline*/to_cusz_pipelinetype(pipeline_type),
        /*predictor*/cusz_custom_predictor{to_cusz_predictor_type(predictor_type), predictor_anchor, predictor_nondestructive},
        /*quantization*/cusz_custom_quantization{quantization_radius, quantization_delayed},
        /*codec*/cusz_custom_codec{to_cusz_codec_type(codec_type), codec_variable_length, codec_presumed_density},
        /*huffman_codec*/cusz_custom_huffman_codec{to_cusz_booktype(huffman_codec_booktype), to_cusz_execution_type(huffman_execution_type), to_cusz_coding_type(huffman_coding), huffman_booklen, huffman_coarse_pardeg}
    };

    std::vector<size_t> norm_dims = input->normalized_dims();
    norm_dims.resize(4);
    std::replace(norm_dims.begin(), norm_dims.end(), 0, 1);

    cusz_compressor* comp       = cusz_create(framework, to_cusz_datatype(input->dtype()));
    cusz_config*     config     = new cusz_config{err_bnd, to_cusz_mode(eb_mode)};
    cusz_len         uncomp_len = cusz_len{{norm_dims.at(0)}, {norm_dims.at(1)}, {norm_dims.at(2)}, {norm_dims.at(3)}, cusz_len_factor};
    auto ec = cusz_compress(
        comp, config, d_uncompressed, uncomp_len, &exposed_compressed, &compressed_len, &header,
        (void*)&compress_timerecord, stream);

    if(ec != CUSZ_SUCCESS) {
      cudaFree(d_uncompressed);
      return cusz_error(ec);
    }

    if(!output->has_data()) {
      uint8_t* host_output;
      cudaMallocHost(&host_output, sizeof(uint8_t)*compressed_len + sizeof(cusz_header));
      *output = pressio_data::move(pressio_byte_dtype, host_output, {compressed_len + sizeof(cusz_header)}, [](void* ptr, void*){
          cudaFreeHost(ptr);
          }, nullptr);
    }


    memcpy(output->data(), &header, sizeof(cusz_header));
    cudaMemcpy(static_cast<uint8_t*>(output->data())+sizeof(cusz_header), exposed_compressed, compressed_len, cudaMemcpyDeviceToHost);
    lp_check_cuda_error(cudaStreamDestroy(stream));
    lp_check_cuda_error(cudaFree(d_uncompressed));


    return 0;
  }

  template<class T>
  int decompress_typed(pressio_data const* input, pressio_data* output) {
    cudaStream_t stream;
    lp_check_cuda_error(cudaStreamCreate(&stream));

    T* d_decompressed;
    uint8_t* d_compressed;
    lp_check_cuda_error(cudaMalloc(&d_compressed, input->size_in_bytes() - sizeof(cusz_header)));
    lp_check_cuda_error(cudaMalloc(&d_decompressed, output->size_in_bytes()));
    lp_check_cuda_error(cudaMemcpy(d_compressed, static_cast<uint8_t*>(input->data()) + sizeof(cusz_header), input->size_in_bytes() - sizeof(cusz_header), cudaMemcpyHostToDevice));

    cusz_framework* framework = new cusz_custom_framework{
        /*datatype*/to_cusz_datatype(output->dtype()),
        /*pipeline*/to_cusz_pipelinetype(pipeline_type),
        /*predictor*/cusz_custom_predictor{to_cusz_predictor_type(predictor_type), predictor_anchor, predictor_nondestructive},
        /*quantization*/cusz_custom_quantization{quantization_radius, quantization_delayed},
        /*codec*/cusz_custom_codec{to_cusz_codec_type(codec_type), codec_variable_length, codec_presumed_density},
        /*huffman_codec*/cusz_custom_huffman_codec{to_cusz_booktype(huffman_codec_booktype), to_cusz_execution_type(huffman_execution_type), to_cusz_coding_type(huffman_coding), huffman_booklen, huffman_coarse_pardeg}
    };

    std::vector<size_t> norm_dims = output->normalized_dims();
    norm_dims.resize(4);
    std::replace(norm_dims.begin(), norm_dims.end(), 0, 1);

    cusz_header header;
    memcpy(&header, input->data(), sizeof(cusz_header));
    cusz_compressor* comp       = cusz_create(framework, to_cusz_datatype(output->dtype()));
    cusz_len         decomp_len = cusz_len{{norm_dims.at(0)}, {norm_dims.at(1)}, {norm_dims.at(2)}, {norm_dims.at(3)}, cusz_len_factor};

    auto ec = cusz_decompress(
        comp, &header, d_compressed, input->size_in_bytes(), d_decompressed, decomp_len,
        (void*)&decompress_timerecord, stream);

    if(ec != CUSZ_SUCCESS) {
      return cusz_error(ec);
    }
    lp_check_cuda_error(cudaMemcpy(static_cast<uint8_t*>(output->data()), d_decompressed, output->size_in_bytes(), cudaMemcpyDeviceToHost));
    lp_check_cuda_error(cudaStreamDestroy(stream));
    lp_check_cuda_error(cudaFree(d_compressed));

    return 0;
  }


  int major_version() const override { return CUSZ_MAJOR_VERSION; }
  int minor_version() const override { return CUSZ_MINOR_VERSION; }
  int patch_version() const override { return CUSZ_PATCH_VERSION; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "cusz"; }

  pressio_options get_metrics_results_impl() const override {
    pressio_options metrics;
    for (auto const& i : compress_timerecord) {
      set(metrics, std::string("cusz:compress_") + std::get<0>(i), std::get<1>(i));
    }
    for (auto const& i : decompress_timerecord) {
      set(metrics, std::string("cusz:decompress_") + std::get<0>(i), std::get<1>(i));
    }

    return metrics;
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<cusz_compressor_plugin>(*this);
  }



  double err_bnd = 1e-5;
  std::string eb_mode = "abs";
  cusz::TimeRecord compress_timerecord;
  cusz::TimeRecord decompress_timerecord;

  std::string pipeline_type = "auto";
  float cusz_len_factor = 1.03;

  bool predictor_anchor = false;
  bool predictor_nondestructive = false;
  std::string predictor_type = "lorenzoi";

  int quantization_radius = 512;
  bool quantization_delayed = false;

  std::string codec_type = "huffman";
  bool codec_variable_length = false;
  float codec_presumed_density = 0;

  std::string huffman_codec_booktype = "tree";
  std::string huffman_execution_type = "device";
  std::string huffman_coding = "coarse";
  int huffman_booklen = 0;
  int huffman_coarse_pardeg = 0;
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "cusz", []() {
  return compat::make_unique<cusz_compressor_plugin>();
});

} }
