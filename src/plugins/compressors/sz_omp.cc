#include <libpressio_ext/cpp/compressor.h>
#include <libpressio_ext/cpp/pressio.h>
#include <std_compat/memory.h>
#include <sz/sz.h>
#include <sz/sz_omp.h>
#include <sstream>
#include "sz_common.h"
#include "iless.h"

static std::map<std::string, int, iless> const sz_omp_mode_str_to_code {
    {"abs", ABS},
};


class sz_omp: public libpressio_compressor_plugin {
  public:
  sz_omp(std::shared_ptr<sz_init_handle>&& init_handle): init_handle(init_handle) {}

  private:
  pressio_options get_options_impl() const override {
    std::shared_lock<std::shared_mutex> lock(init_handle->sz_init_lock);
    struct pressio_options options;
    set_type(options, "sz_omp:config_file", pressio_option_charptr_type);
    set_type(options, "sz_omp:config_struct", pressio_option_userptr_type);
#if PRESSIO_SZ_VERSION_GREATEREQ(2,1,9,0)
    set(options, "sz_omp:protect_value_range", confparams_cpr->protectValueRange);
#endif
    set(options, "sz_omp:max_quant_intervals", confparams_cpr->max_quant_intervals);
    set(options, "sz_omp:quantization_intervals", confparams_cpr->quantization_intervals);
    set(options, "sz_omp:sol_id", confparams_cpr->sol_ID);
    set(options, "sz_omp:lossless_compressor", confparams_cpr->losslessCompressor);
    set(options, "sz_omp:sample_distance", confparams_cpr->sampleDistance);
    set(options, "sz_omp:pred_threshold", confparams_cpr->predThreshold);
    set(options, "sz_omp:sz_mode", confparams_cpr->szMode);
    set(options, "sz_omp:gzip_mode", confparams_cpr->gzipMode);
    set(options, "sz_omp:error_bound_mode", confparams_cpr->errorBoundMode);
    set_type(options, "sz_omp:error_bound_mode_str", pressio_option_charptr_type);
	  set(options, "sz_omp:abs_err_bound", confparams_cpr->absErrBound);
	  set(options, "sz_omp:rel_err_bound", confparams_cpr->relBoundRatio);
	  set(options, "sz_omp:psnr_err_bound", confparams_cpr->psnr);
	  set(options, "sz_omp:pw_rel_err_bound", confparams_cpr->pw_relBoundRatio);
	  set(options, "sz_omp:segment_size", confparams_cpr->segment_size);
	  set(options, "sz_omp:snapshot_cmpr_step", confparams_cpr->snapshotCmprStep);
	  set(options, "sz_omp:accelerate_pw_rel_compression", confparams_cpr->accelerate_pw_rel_compression);
	  set_type(options, "sz_omp:prediction_mode", pressio_option_int32_type);
    set_type(options, "sz_omp:data_type", pressio_option_double_type);
#ifdef HAVE_RANDOMACCESS
    set(options, "sz_omp:random_access", confparams_cpr->randomAccess);
#endif
    return options;
  }
  pressio_options get_documentation_impl() const override {
    struct pressio_options options;
    set(options, "pressio:description", R"(SZ is an error bounded lossy compressor that uses prediction 
      based methods to compress data. This is SZ's native multi-threaded compression support)");
    set(options, "sz_omp:random_access_enabled", "true if SZ was compiled in random access mode");
    set(options, "sz_omp:timecmpr", "true if SZ if SZ is compiled in time based compression mode");
    set(options, "sz_omp:pastri", "true if PASTRI mode was built");
    set(options, "sz_omp:write_stats", "true if SZ is compiled with compression statistics support");
    set(options, "sz_omp:abs_err_bound", "the absolute error bound ");
    set(options, "sz_omp:accelerate_pw_rel_compression", "trade compression ratio for a faster pw_rel compression");
    set(options, "sz_omp:config_file", "filepath passed to SZ_Init()");
    set(options, "sz_omp:config_struct", "structure passed to SZ_Init_Params()" );
    set(options, "sz_omp:data_type", "an internal option to control compression");
    set(options, "sz_omp:error_bound_mode", "integer code used to determine error bound mode");
    set(options, "sz_omp:error_bound_mode_str", "human readable string to set the error bound mode");
    set(options, "sz_omp:gzip_mode", "Which mode to pass to GZIP when used");
    set(options, "sz_omp:lossless_compressor", "Which lossless compressor to use for stage 4");
    set(options, "sz_omp:max_quant_intervals", "the maximum number of quantization intervals");
    set(options, "sz_omp:pred_threshold", "an internal option used to control compression");
    set(options, "sz_omp:prediction_mode", "an internal option used to control compression");
    set(options, "sz_omp:psnr_err_bound", "the bound on the error in the PSNR");
    set(options, "sz_omp:pw_rel_err_bound", "the bound on the pointwise relative error");
    set(options, "sz_omp:quantization_intervals", "the number of quantization intervals to use, 0 means automatic");
    set(options, "sz_omp:random_access", "internal options to use random access mode when compiled in");
    set(options, "sz_omp:rel_err_bound", "the value range relative error bound mode");
    set(options, "sz_omp:sample_distance", "internal option used to control compression");
    set(options, "sz_omp:segment_size", "internal option used to control compression. number of points in each segement for pw_relBoundRatio");
    set(options, "sz_omp:snapshot_cmpr_step", "the frequency of preforming single snapshot based compression in time based compression");
    set(options, "sz_omp:sol_id", "an internal option use d to control compression");
    set(options, "sz_omp:sz_mode", "SZ Mode either SZ_BEST_COMPRESSION or SZ_BEST_SPEED");
    set(options, "sz_omp:user_params", "arguments passed to the application specific mode of SZ. Use in conjunction with sz_omp:app");
    set(options, "sz_omp:protect_value_range", "should the value range be preserved during compression");
    return options;
  }
  pressio_options get_configuration_impl() const override {
    pressio_options options;
    set(options, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_serialized));
    set(options, "pressio:stability", "experimental");
#ifdef HAVE_RANDOMACCESS
    set(options, "sz_omp:random_access_enabled", 1u);
#else
    set(options, "sz_omp:random_access_enabled", 0u);
#endif
#ifdef HAVE_TIMECMPR
    set(options, "sz_omp:timecmpr", 1u);
#else
    set(options, "sz_omp:timecmpr", 0u);
#endif
#ifdef HAVE_PASTRI
    set(options, "sz_omp:pastri", 1u);
#else
    set(options, "sz_omp:pastri", 0u);
#endif
#ifdef HAVE_WRITESTATS
    set(options, "sz_omp:write_stats", 1u);
#else
    set(options, "sz_omp:write_stats", 0u);
#endif


    std::vector<std::string> vs;
    std::transform(
        std::begin(sz_omp_mode_str_to_code),
        std::end(sz_omp_mode_str_to_code),
        std::back_inserter(vs),
        [](typename decltype(sz_omp_mode_str_to_code)::const_reference m){ return m.first; });
    set(options, "sz_omp:error_bound_mode_str", vs);
    return options;
  }
  int set_options_impl(const pressio_options &options) override {
    struct sz_params* sz_param;
    std::string config_file;
    if(get(options, "sz:config_file", &config_file) == pressio_options_key_set) {
      SZ_Finalize();
      SZ_Init(config_file.c_str());
    } else if (get(options, "sz:config_struct", (void**)&sz_param) == pressio_options_key_set) {
      SZ_Finalize();
      SZ_Init_Params(sz_param);
    }

#if PRESSIO_SZ_VERSION_GREATEREQ(2,1,9,0)
    get(options, "sz:protect_value_range", &confparams_cpr->protectValueRange);
#endif
    get(options, "sz:max_quant_intervals", &confparams_cpr->max_quant_intervals);
    get(options, "sz:quantization_intervals", &confparams_cpr->quantization_intervals);
    get(options, "sz:sol_id", &confparams_cpr->sol_ID);
    get(options, "sz:lossless_compressor", &confparams_cpr->losslessCompressor);
    get(options, "sz:sample_distance", &confparams_cpr->sampleDistance);
    get(options, "sz:pred_threshold", &confparams_cpr->predThreshold);
    get(options, "sz:sz_mode", &confparams_cpr->szMode);
    get(options, "sz:gzip_mode", &confparams_cpr->gzipMode);

    std::string error_bound_mode_str;
    if(get(options, "sz:error_bound_mode_str", &error_bound_mode_str) == pressio_options_key_set) {
      auto key = sz_omp_mode_str_to_code.find(error_bound_mode_str);
      if(key != sz_omp_mode_str_to_code.end()) {
        confparams_cpr->errorBoundMode = key->second;
      }
    } else { 
      get(options, "sz:error_bound_mode", &confparams_cpr->errorBoundMode ); 
    }
    get(options, "sz:abs_err_bound", &confparams_cpr->absErrBound);
    get(options, "sz:rel_err_bound", &confparams_cpr->relBoundRatio);
    get(options, "sz:psnr_err_bound", &confparams_cpr->psnr);
    get(options, "sz:pw_rel_err_bound", &confparams_cpr->pw_relBoundRatio);
    get(options, "sz:segment_size", &confparams_cpr->segment_size);
    get(options, "sz:snapshot_cmpr_step", &confparams_cpr->snapshotCmprStep);
    get(options, "sz:prediction_mode", &confparams_cpr->predictionMode);
    get(options, "sz:accelerate_pw_rel_compression", &confparams_cpr->accelerate_pw_rel_compression);
    get(options, "sz:data_type", &confparams_cpr->dataType);
#ifdef HAVE_RANDOMACCESS
    get(options, "sz:random_access", &confparams_cpr->randomAccess);
#endif
    return 0;
  }
  int compress_impl(const pressio_data *input, struct pressio_data *output) override {
    if(input->dtype() != pressio_float_dtype) {
      return set_error(1, "only floating point data is supported");
    }
    if(input->num_dimensions() != 3) {
      return set_error(2, "only 3d data is supported");
    }
    std::unique_lock<std::shared_mutex> lock(init_handle->sz_init_lock);
    confparams_cpr->dataType = SZ_FLOAT;
    size_t r1 = pressio_data_get_dimension(input, 0);
    size_t r2 = pressio_data_get_dimension(input, 1);
    size_t r3 = pressio_data_get_dimension(input, 2);
    size_t outSize;
    //expected to be set to float, but the user might have other uses, so save it first here before
    //making changes
    auto old_dtype = confparams_cpr->dataType;
    confparams_cpr->dataType = SZ_FLOAT;

    confparams_cpr->dataType = SZ_FLOAT;
    unsigned char* bytes;
    switch(input->num_dimensions()) {
        case 3:
					bytes = SZ_compress_float_3D_MDQ_openmp(static_cast<float*>(input->data()), r3, r2, r1, static_cast<float>(confparams_cpr->absErrBound), &outSize);
          break;
    }
    confparams_cpr->dataType = old_dtype;
    if(bytes != nullptr) {
      *output = pressio_data::move(pressio_byte_dtype, bytes, 1, &outSize, pressio_data_libc_free_fn, nullptr);
    } else {
      return set_error(3, "compression failed");
    }

    return 0;
  }
  int decompress_impl(const pressio_data *input, struct pressio_data *output) override {
    if(output->dtype() != pressio_float_dtype) {
      return set_error(1, "only floating point data is supported");
    }
    if(output->num_dimensions() != 3) {
      return set_error(2, "only 3d data is supported");
    }
    std::unique_lock<std::shared_mutex> lock(init_handle->sz_init_lock);

    //expected to be set to float, but the user might have other uses, so save it first here before
    //making changes
    auto old_dtype = confparams_cpr->dataType;
    confparams_cpr->dataType = SZ_FLOAT;

    unsigned char* bytes = static_cast<unsigned char*>(input->data());
    float* output_data = NULL;
    size_t r1 = pressio_data_get_dimension(output, 0);
    size_t r2 = pressio_data_get_dimension(output, 1);
    size_t r3 = pressio_data_get_dimension(output, 2);
    switch(output->num_dimensions()) {
      case 3:
					decompressDataSeries_float_3D_openmp(&output_data, r3, r2, r1, bytes + 1+3+MetaDataByteLength);
          break;
    }
    confparams_cpr->dataType = old_dtype;
    if(output_data != nullptr) {
      *output = pressio_data::move(pressio_float_dtype, output_data, output->dimensions(), pressio_data_libc_free_fn, nullptr);
    } else {
      return set_error(3, "compression failed");
    }

    return 0;
  }
  const char* prefix() const override { return "sz_omp"; }
  const char* version() const override { 
    const static std::string version_str = [this]{
      std::stringstream ss;
      ss << major_version() << '.' << minor_version() << '.' << patch_version();
      return ss.str();
    }();
    return version_str.c_str();
  }
  virtual std::shared_ptr<libpressio_compressor_plugin> clone() override {
    return compat::make_unique<sz_omp>(*this);
  }

  std::shared_ptr<sz_init_handle> init_handle;
};

static pressio_register sz_omp_register(
    compressor_plugins(),
    "sz_omp",
    []{
      return compat::make_unique<sz_omp>(pressio_get_sz_init_handle());
    }
    );

