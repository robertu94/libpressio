#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "iless.h"

#include <QoZ/api/sz.hpp>


namespace libpressio { namespace compressors { namespace qoz_ns {


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
        pressio_new_free_fn<QoZ::uchar>(),
        nullptr
        );
  }
  pressio_data operator()(bool* , bool*) {
        throw std::runtime_error("unsupported type bool");
  }
  pressio_data const& input_data;
  QoZ::Config& config;
  std::vector<size_t> const& reg_dims;
};

  
struct sz3_option_maps {
  std::map<std::string, QoZ::EB, iless> error_bounds;
  std::map<std::string, QoZ::ALGO, iless> algo;
  //std::map<std::string, QoZ::INTERP_ALGO, iless> interp_algo; //hided due to code update
  std::map<std::string, QoZ::TUNING_TARGET, iless> tuning_options;
  sz3_option_maps() {
    for (size_t i = 0; i < std::size(QoZ::EB_STR); ++i) {
      error_bounds[QoZ::EB_STR[i]] = QoZ::EB_OPTIONS[i];
    }
    for (size_t i = 0; i < std::size(QoZ::ALGO_STR); ++i) {
      algo[QoZ::ALGO_STR[i]] = QoZ::ALGO_OPTIONS[i];
    }
    /*
    for (size_t i = 0; i < std::size(QoZ::INTERP_ALGO_STR); ++i) {
      interp_algo[QoZ::INTERP_ALGO_STR[i]] = QoZ::INTERP_ALGO_OPTIONS[i];
    }
    *///hided due to code update
    for (size_t i = 0; i < std::size(QoZ::TUNING_TARGET_STR); ++i) {
      tuning_options[QoZ::TUNING_TARGET_STR[i]] = QoZ::TUNING_TARGET_OPTIONS[i];
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
    config.errorBoundMode = QoZ::EB_ABS;//Should EB_REL be better?
    config.QoZ = 3;//updated
    config.verbose = 0;
  }

  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    if(config.errorBoundMode == QoZ::EB_ABS) {
      set(options, "pressio:abs", config.absErrorBound);
    } else {
      set_type(options, "pressio:abs", pressio_option_double_type);
    }
    if(config.errorBoundMode == QoZ::EB_REL) {
      set(options, "pressio:rel", config.relErrorBound);
    } else {
      set_type(options, "pressio:rel", pressio_option_double_type);
    }
    set(options, "qoz:abs_error_bound", config.absErrorBound);
    set(options, "qoz:rel_error_bound", config.relErrorBound);
    set(options, "qoz:psnr_error_bound", config.psnrErrorBound);
    set(options, "qoz:l2_norm_error_bound", config.l2normErrorBound);
    set(options, "qoz:error_bound_mode", config.errorBoundMode);
    set(options, "qoz:algorithm", config.cmprAlgo);
    set(options, "qoz:lorenzo", config.lorenzo);
    set(options, "qoz:lorenzo2", config.lorenzo2);
    set(options, "qoz:regression", config.regression);
    set(options, "qoz:regression2", config.regression2);
    set(options, "qoz:openmp", config.openmp);
    set(options, "qoz:lossless", config.lossless);
    set(options, "qoz:encoder", config.encoder);
    //set(options, "qoz:interp_algo", config.interpAlgo); //hided due to code update
    //set(options, "qoz:interp_direction", config.interpDirection); //hided due to code update
    set(options, "qoz:interp_block_size", config.interpBlockSize);
    set(options, "qoz:quant_bin_size", config.quantbinCnt);
    set(options, "qoz:stride", config.stride);
    set(options, "qoz:pred_dim", config.pred_dim);
    set(options, "qoz:qoz_level", config.QoZ);//Updated the option name.
    set(options, "qoz:maxstep", config.maxStep);
    set(options, "qoz:test_lorenzo", config.testLorenzo);
    set(options, "qoz:tuning_target", config.tuningTarget);//fixed a typo. turning to tuning
    set_type(options, "qoz:tuning_target_str", pressio_option_charptr_type);
    set_type(options, "qoz:error_bound_mode_str", pressio_option_charptr_type);
    //set_type(options, "qoz:intrep_algo_str", pressio_option_charptr_type);//hided due to code update
    set_type(options, "qoz:algorithm_str", pressio_option_charptr_type);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    set(options, "qoz:error_bound_mode_str", keys(sz3_options().error_bounds));
    //set(options, "qoz:intrep_algo_str", keys(sz3_options().interp_algo));//hided due to code update
    set(options, "qoz:tuning_target_str", keys(sz3_options().tuning_options));
    set(options, "qoz:algorithm_str", keys(sz3_options().algo));
    
        std::vector<std::string> invalidations {"qoz:abs_error_bound", "qoz:rel_error_bound", "qoz:psnr_error_bound", "qoz:l2_norm_error_bound", "qoz:error_bound_mode", "qoz:algorithm", "qoz:lorenzo", "qoz:lorenzo2", "qoz:regression", "qoz:regression2", "qoz:openmp", "qoz:lossless", "qoz:encoder", "qoz:interp_algo", "qoz:interp_direction", "qoz:interp_block_size", "qoz:quant_bin_size", "qoz:stride", "qoz:pred_dim", "qoz:use_qoz", "qoz:maxstep", "qoz:test_lorenzo", "qoz:tuning_target", "pressio:abs", "pressio:rel", "qoz:error_bound_mode_str", "qoz:intrep_algo_str", "qoz:algorithm_str", "qoz:tuning_target_str"}; 
        std::vector<pressio_configurable const*> invalidation_children {}; 
        
        set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
        set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
        set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));

    
        set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{"pressio:abs", "pressio:rel"}));

    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(QoZ is the evolution of SZ3 a modular compression framework)");
    set(options, "qoz:abs_error_bound", "absolute error bound");
    set(options, "qoz:rel_error_bound", "value range relative error bound");
    set(options, "qoz:psnr_error_bound", "psnr error bound");
    set(options, "qoz:l2_norm_error_bound", "l2 norm error bound");
    set(options, "qoz:error_bound_mode", "error bound mode to apply");
    set(options, "qoz:algorithm", "compression algorithm");
    set(options, "qoz:lorenzo", "use the lorenzo predictor");
    set(options, "qoz:lorenzo2", "use the 2-level lorenzo predictor");
    set(options, "qoz:regression", "use the regression predictor");
    set(options, "qoz:regression2", "use the 2nd order regression predictor");
    set(options, "qoz:openmp", "use openmp parallelization");
    set(options, "qoz:lossless", "lossless compression method to apply; 1 bypass lossless, 1 zstd");
    set(options, "qoz:encoder", "which encoder to use, 0 skip encoder, 1 huffman, 2 arithmatic");
    //set(options, "qoz:interp_algo", "which intrepolation algorithm to use");
   // set(options, "qoz:interp_direction", "which interpolation direction to use");
    set(options, "qoz:interp_block_size", "what block size to use for interpolation to use");
    set(options, "qoz:quant_bin_size", "number of quantization bins");
    set(options, "qoz:stride", "stride between items");
    set(options, "qoz:pred_dim", "prediction dimension");
    set(options, "qoz:algorithm_str", "compression algorithm");
    set(options, "qoz:error_bound_mode_str", "error bound");
    //set(options, "qoz:intrep_algo_str", "interpolation algorithm mode");
    set(options, "qoz:qoz_level", "optimization level of QoZ compression. 0/1/2/3/4 are available. 0 is SZ3, 1 is QoZ1, 2 or 3 are recommended for QoZ2.");
    set(options, "qoz:maxstep", "set the maximum step size");
    set(options, "qoz:test_lorenzo", "test lorenzo predictor and use it when it is better");
    set(options, "qoz:tuning_target", "the tuning target for quality");
    set(options, "qoz:tuning_target_str", "the tuning target as a string");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    if(get(options, "pressio:abs", &config.absErrorBound) == pressio_options_key_set) {
      config.errorBoundMode = QoZ::EB_ABS;
    } 
    if(get(options, "pressio:rel", &config.relErrorBound) == pressio_options_key_set) {
      config.errorBoundMode = QoZ::EB_REL;
    } 
    get(options, "qoz:abs_error_bound", &config.absErrorBound);
    get(options, "qoz:rel_error_bound", &config.relErrorBound);
    get(options, "qoz:psnr_error_bound", &config.psnrErrorBound);
    get(options, "qoz:l2_norm_error_bound", &config.l2normErrorBound);
    get(options, "qoz:error_bound_mode", &config.errorBoundMode);
    get(options, "qoz:algorithm", &config.cmprAlgo);
    get(options, "qoz:lorenzo", &config.lorenzo);
    get(options, "qoz:lorenzo2", &config.lorenzo2);
    get(options, "qoz:regression", &config.regression);
    get(options, "qoz:regression2", &config.regression2);
    get(options, "qoz:openmp", &config.openmp);
    get(options, "qoz:lossless", &config.lossless);
    get(options, "qoz:encoder", &config.encoder);
    //get(options, "qoz:interp_algo", &config.interpAlgo);
    //get(options, "qoz:interp_direction", &config.interpDirection);
    get(options, "qoz:interp_block_size", &config.interpBlockSize);
    get(options, "qoz:quant_bin_size", &config.quantbinCnt);
    get(options, "qoz:stride", &config.stride);
    get(options, "qoz:pred_dim", &config.pred_dim);
    get(options, "qoz:qoz_level", &config.QoZ);
    get(options, "qoz:maxstep", &config.maxStep);
    get(options, "qoz:test_lorenzo", &config.testLorenzo);
    get(options, "qoz:tuning_target", &config.tuningTarget);
    std::string tmp;
    try {
      if(get(options, "qoz:error_bound_mode_str", &tmp) == pressio_options_key_set) {
        config.errorBoundMode = sz3_options().error_bounds.at(tmp);
      }
      /*
      if(get(options, "qoz:intrep_algo_str", &tmp) == pressio_options_key_set) {
        config.interpAlgo = sz3_options().interp_algo.at(tmp);
      }*/
      if(get(options, "qoz:algorithm_str", &tmp) == pressio_options_key_set) {
        config.cmprAlgo = sz3_options().algo.at(tmp);
      }
      if(get(options, "qoz:tuning_target_str", &tmp) == pressio_options_key_set ) {
        config.tuningTarget = sz3_options().tuning_options.at(tmp);
      }
    } catch(std::out_of_range const& ex) {
      return set_error(1, ex.what());
    }
    return 0;
  }

  int compress_impl(const pressio_data* real_input,
                    struct pressio_data* output) override
  {
    pressio_data input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
    auto reg_dims = input.normalized_dims();
    std::reverse(reg_dims.begin(), reg_dims.end());
    config.dims = reg_dims;
    if(reg_dims.size() > std::numeric_limits<char>::max()) {
      set_error(-1, "overflow of sz3 N parameter");
    }
    *output = pressio_data_for_each<pressio_data>(input, impl_compress{input, config, reg_dims});
    return 0;
  }

  int decompress_impl(const pressio_data* real_input,
                      struct pressio_data* output) override
  {
    pressio_data input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);

    switch(output->dtype()) {
      case pressio_float_dtype:
        {
          auto decData = static_cast<float*>(output->data());
          SZ_decompress(config, static_cast<char*>(input.data()), input.num_elements(), decData);
          break;
        }
      case pressio_double_dtype:
        {
          auto decData = static_cast<double*>(output->data());
          SZ_decompress(config, static_cast<char*>(input.data()), input.num_elements(), decData);
          break;
        }
      case pressio_int8_dtype:
        {
          auto decData = static_cast<int8_t*>(output->data());
          SZ_decompress(config, static_cast<char*>(input.data()), input.num_elements(), decData);
          break;
        }
      case pressio_int16_dtype:
        {
          auto decData = static_cast<int16_t*>(output->data());
          SZ_decompress(config, static_cast<char*>(input.data()), input.num_elements(), decData);
          break;
        }
      case pressio_int32_dtype:
        {
          auto decData = static_cast<int32_t*>(output->data());
          SZ_decompress(config, static_cast<char*>(input.data()), input.num_elements(), decData);
          break;
        }
      case pressio_int64_dtype:
        {
          auto decData = static_cast<int64_t*>(output->data());
          SZ_decompress(config, static_cast<char*>(input.data()), input.num_elements(), decData);
          break;
        }
      case pressio_uint8_dtype:
        {
          auto decData = static_cast<uint8_t*>(output->data());
          SZ_decompress(config, static_cast<char*>(input.data()), input.num_elements(), decData);
          break;
        }
      case pressio_uint16_dtype:
        {
          auto decData = static_cast<uint16_t*>(output->data());
          SZ_decompress(config, static_cast<char*>(input.data()), input.num_elements(), decData);
          break;
        }
      case pressio_uint32_dtype:
        {
          auto decData = static_cast<uint32_t*>(output->data());
          SZ_decompress(config, static_cast<char*>(input.data()), input.num_elements(), decData);
          break;
        }
      case pressio_uint64_dtype:
        {
          auto decData = static_cast<uint64_t*>(output->data());
          SZ_decompress(config, static_cast<char*>(input.data()), input.num_elements(), decData);
          break;
        }
      default:
        return set_error(1, "unsupported type");
    }
    return 0;
  }

  int major_version() const override { return QoZ_VER_MAJOR; }
  int minor_version() const override { return QoZ_VER_MINOR; }
  int patch_version() const override { return QoZ_VER_PATCH; }
  const char* version() const override { return QoZ_VER; }
  const char* prefix() const override { return "qoz"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<sz3_compressor_plugin>(*this);
  }

  QoZ::Config config;
};

pressio_register registration(compressor_plugins(), "qoz", []() {
  return compat::make_unique<sz3_compressor_plugin>();
});

} }}
