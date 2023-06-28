#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "iless.h"
#include "cleanup.h"

#include <SZ3/api/sz.hpp>


namespace libpressio { namespace sz_interp_ns {


struct impl_compress{
  template <class T>
  typename std::enable_if<!std::is_same<T, bool>::value,pressio_data>::type operator()(T* in_data, T*) {
    config.N = static_cast<char>(reg_dims.size());
    config.num = std::accumulate(reg_dims.begin(), reg_dims.end(), (size_t) 1, compat::multiplies<>());
    config.blockSize = (config.N == 1 ? 128 : (config.N == 2 ? 16 : 6));
    config.pred_dim = static_cast<unsigned char>(config.N);
    config.stride = config.blockSize;
    size_t outSize = 0;
    char *compressedData = SZ_compress(config, in_data, outSize);
    return pressio_data::move(
        pressio_byte_dtype,
        compressedData,
        {outSize},
        pressio_new_free_fn<SZ::uchar>(),
        nullptr
        );
  }
  pressio_data operator()(bool* , bool*) {
        throw std::runtime_error("unsupported type bool");
  }
  pressio_data const& input_data;
  SZ::Config& config;
  std::vector<size_t> const& reg_dims;
};

  
struct sz3_option_maps {
  std::map<std::string, SZ::EB, iless> error_bounds;
  std::map<std::string, SZ::ALGO, iless> algo;
  std::map<std::string, SZ::INTERP_ALGO, iless> interp_algo;
  sz3_option_maps() {
    for (size_t i = 0; i < std::size(SZ::EB_STR); ++i) {
      error_bounds[SZ::EB_STR[i]] = SZ::EB_OPTIONS[i];
    }
    for (size_t i = 0; i < std::size(SZ::ALGO_STR); ++i) {
      algo[SZ::ALGO_STR[i]] = SZ::ALGO_OPTIONS[i];
    }
    for (size_t i = 0; i < std::size(SZ::INTERP_ALGO_STR); ++i) {
      interp_algo[SZ::INTERP_ALGO_STR[i]] = SZ::INTERP_ALGO_OPTIONS[i];
    }
  }
};
sz3_option_maps const& sz3_options() {
  static sz3_option_maps maps;
  return maps;
}
template <class K, class V, class Sort>
std::vector<K> keys(std::map<K,V,Sort> const& map) {
  std::vector<K> k;
  k.reserve(map.size());
  for (auto const& i : map) {
    k.emplace_back(i.first);
  }
  return k;
}

class sz3_compressor_plugin : public libpressio_compressor_plugin {
public:
  sz3_compressor_plugin() {
    config.absErrorBound = 1e-6;
    config.errorBoundMode = SZ::EB_ABS;
  }

  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    if(config.errorBoundMode == SZ::EB_ABS) {
      set(options, "pressio:abs", config.absErrorBound);
    } else {
      set_type(options, "pressio:abs", pressio_option_double_type);
    }
    if(config.errorBoundMode == SZ::EB_REL) {
      set(options, "pressio:rel", config.relErrorBound);
    } else {
      set_type(options, "pressio:rel", pressio_option_double_type);
    }
    set(options, "pressio:nthreads", nthreads);
    set(options, "sz3:abs_error_bound", config.absErrorBound);
    set(options, "sz3:rel_error_bound", config.relErrorBound);
    set(options, "sz3:psnr_error_bound", config.psnrErrorBound);
    set(options, "sz3:l2_norm_error_bound", config.l2normErrorBound);
    set(options, "sz3:error_bound_mode", config.errorBoundMode);
    set(options, "sz3:algorithm", config.cmprAlgo);
    set(options, "sz3:lorenzo", config.lorenzo);
    set(options, "sz3:lorenzo2", config.lorenzo2);
    set(options, "sz3:regression", config.regression);
    set(options, "sz3:regression2", config.regression2);
    set(options, "sz3:openmp", config.openmp);
    set(options, "sz3:lossless", config.lossless);
    set(options, "sz3:encoder", config.encoder);
    set(options, "sz3:interp_algo", config.interpAlgo);
    set(options, "sz3:interp_direction", config.interpDirection);
    set(options, "sz3:interp_block_size", config.interpBlockSize);
    set(options, "sz3:quant_bin_size", config.quantbinCnt);
    set(options, "sz3:stride", config.stride);
    set(options, "sz3:pred_dim", config.pred_dim);
    set_type(options, "sz3:error_bound_mode_str", pressio_option_charptr_type);
    set_type(options, "sz3:intrep_algo_str", pressio_option_charptr_type);
    set_type(options, "sz3:algorithm_str", pressio_option_charptr_type);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    set(options, "sz3:error_bound_mode_str", keys(sz3_options().error_bounds));
    set(options, "sz3:intrep_algo_str", keys(sz3_options().interp_algo));
    set(options, "sz3:algorithm_str", keys(sz3_options().algo));
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(SZ3 is a modular compressor framework)");
    set(options, "sz3:abs_error_bound", "absolute error bound");
    set(options, "sz3:rel_error_bound", "value range relative error bound");
    set(options, "sz3:psnr_error_bound", "psnr error bound");
    set(options, "sz3:l2_norm_error_bound", "l2 norm error bound");
    set(options, "sz3:error_bound_mode", "error bound mode to apply");
    set(options, "sz3:algorithm", "compression algorithm");
    set(options, "sz3:lorenzo", "use the lorenzo predictor");
    set(options, "sz3:lorenzo2", "use the 2-level lorenzo predictor");
    set(options, "sz3:regression", "use the regression predictor");
    set(options, "sz3:regression2", "use the 2nd order regression predictor");
    set(options, "sz3:openmp", "use openmp parallelization");
    set(options, "sz3:lossless", "lossless compression method to apply; 1 bypass lossless, 1 zstd");
    set(options, "sz3:encoder", "which encoder to use, 0 skip encoder, 1 huffman, 2 arithmatic");
    set(options, "sz3:interp_algo", "which intrepolation algorithm to use");
    set(options, "sz3:interp_direction", "which interpolation direction to use");
    set(options, "sz3:interp_block_size", "what block size to use for interpolation to use");
    set(options, "sz3:quant_bin_size", "number of quantization bins");
    set(options, "sz3:stride", "stride between items");
    set(options, "sz3:pred_dim", "prediction dimension");
    set(options, "sz3:algorithm_str", "compression algorithm");
    set(options, "sz3:error_bound_mode_str", "error bound");
    set(options, "sz3:intrep_algo_str", "interpolation algorithm mode");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    if(get(options, "pressio:abs", &config.absErrorBound) == pressio_options_key_set) {
      config.errorBoundMode = SZ::EB_ABS;
    } 
    if(get(options, "pressio:rel", &config.relErrorBound) == pressio_options_key_set) {
      config.errorBoundMode = SZ::EB_REL;
    } 
    uint32_t tmp_nthreads;
    if(get(options, "pressio:nthreads", &tmp_nthreads) == pressio_options_key_set) {
        if(tmp_nthreads > 1) {
            nthreads = tmp_nthreads;
            config.openmp = true;
        } else if(tmp_nthreads == 1) {
            nthreads = tmp_nthreads;
            config.openmp = false;
        } else {
            return set_error(1, "unsupported nthreads");
        }
    }
    get(options, "sz3:abs_error_bound", &config.absErrorBound);
    get(options, "sz3:rel_error_bound", &config.relErrorBound);
    get(options, "sz3:psnr_error_bound", &config.psnrErrorBound);
    get(options, "sz3:l2_norm_error_bound", &config.l2normErrorBound);
    get(options, "sz3:error_bound_mode", &config.errorBoundMode);
    get(options, "sz3:algorithm", &config.cmprAlgo);
    get(options, "sz3:lorenzo", &config.lorenzo);
    get(options, "sz3:lorenzo2", &config.lorenzo2);
    get(options, "sz3:regression", &config.regression);
    get(options, "sz3:regression2", &config.regression2);
    get(options, "sz3:openmp", &config.openmp);
    get(options, "sz3:lossless", &config.lossless);
    get(options, "sz3:encoder", &config.encoder);
    get(options, "sz3:interp_algo", &config.interpAlgo);
    get(options, "sz3:interp_direction", &config.interpDirection);
    get(options, "sz3:interp_block_size", &config.interpBlockSize);
    get(options, "sz3:quant_bin_size", &config.quantbinCnt);
    get(options, "sz3:stride", &config.stride);
    get(options, "sz3:pred_dim", &config.pred_dim);
    std::string tmp;
    try {
      if(get(options, "sz3:error_bound_mode_str", &tmp) == pressio_options_key_set) {
        config.errorBoundMode = sz3_options().error_bounds.at(tmp);
      }
      if(get(options, "sz3:intrep_algo_str", &tmp) == pressio_options_key_set) {
        config.interpAlgo = sz3_options().interp_algo.at(tmp);
      }
      if(get(options, "sz3:algorithm_str", &tmp) == pressio_options_key_set) {
        config.cmprAlgo = sz3_options().algo.at(tmp);
      }
    } catch(std::out_of_range const& ex) {
      return set_error(1, ex.what());
    }
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    cleanup restore_threads;
    if(config.openmp) {
        int32_t old_threads = omp_get_num_threads();
        omp_set_num_threads(static_cast<int32_t>(nthreads));
        restore_threads = [old_threads]{ omp_set_num_threads(old_threads);};
    }

    auto reg_dims = input->normalized_dims();
    std::reverse(reg_dims.begin(), reg_dims.end());
    config.dims = reg_dims;
    if(reg_dims.size() > std::numeric_limits<char>::max()) {
      set_error(-1, "overflow of sz3 N parameter");
    }
    *output = pressio_data_for_each<pressio_data>(*input, impl_compress{*input, config, reg_dims});
    return 0;
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    cleanup restore_threads;
    if(config.openmp) {
        int32_t old_threads = omp_get_num_threads();
        omp_set_num_threads(static_cast<int32_t>(nthreads));
        restore_threads = [old_threads]{ omp_set_num_threads(old_threads);};
    }

    switch(output->dtype()) {
      case pressio_float_dtype:
        {
          auto decData = static_cast<float*>(output->data());
          SZ_decompress(config, static_cast<char*>(input->data()), input->num_elements(), decData);
          break;
        }
      case pressio_double_dtype:
        {
          auto decData = static_cast<double*>(output->data());
          SZ_decompress(config, static_cast<char*>(input->data()), input->num_elements(), decData);
          break;
        }
      case pressio_int8_dtype:
        {
          auto decData = static_cast<int8_t*>(output->data());
          SZ_decompress(config, static_cast<char*>(input->data()), input->num_elements(), decData);
          break;
        }
      case pressio_int16_dtype:
        {
          auto decData = static_cast<int16_t*>(output->data());
          SZ_decompress(config, static_cast<char*>(input->data()), input->num_elements(), decData);
          break;
        }
      case pressio_int32_dtype:
        {
          auto decData = static_cast<int32_t*>(output->data());
          SZ_decompress(config, static_cast<char*>(input->data()), input->num_elements(), decData);
          break;
        }
      case pressio_int64_dtype:
        {
          auto decData = static_cast<int64_t*>(output->data());
          SZ_decompress(config, static_cast<char*>(input->data()), input->num_elements(), decData);
          break;
        }
      case pressio_uint8_dtype:
        {
          auto decData = static_cast<uint8_t*>(output->data());
          SZ_decompress(config, static_cast<char*>(input->data()), input->num_elements(), decData);
          break;
        }
      case pressio_uint16_dtype:
        {
          auto decData = static_cast<uint16_t*>(output->data());
          SZ_decompress(config, static_cast<char*>(input->data()), input->num_elements(), decData);
          break;
        }
      case pressio_uint32_dtype:
        {
          auto decData = static_cast<uint32_t*>(output->data());
          SZ_decompress(config, static_cast<char*>(input->data()), input->num_elements(), decData);
          break;
        }
      case pressio_uint64_dtype:
        {
          auto decData = static_cast<uint64_t*>(output->data());
          SZ_decompress(config, static_cast<char*>(input->data()), input->num_elements(), decData);
          break;
        }
      default:
        return set_error(1, "unsupported type");
    }
    return 0;
  }

  int major_version() const override { return SZ3_VER_MAJOR; }
  int minor_version() const override { return SZ3_VER_MINOR; }
  int patch_version() const override { return SZ3_VER_PATCH; }
  const char* version() const override { return SZ3_VER; }
  const char* prefix() const override { return "sz3"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<sz3_compressor_plugin>(*this);
  }

  uint32_t nthreads = 1;
  SZ::Config config;
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "sz3", []() {
  return compat::make_unique<sz3_compressor_plugin>();
});

} }
