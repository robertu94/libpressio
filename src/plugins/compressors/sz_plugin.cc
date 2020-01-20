#include <memory>
#include <sstream>
#include <cstdlib>

#include <sz/sz.h>

#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/compat/std_compat.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "pressio_option.h"


class sz_plugin: public libpressio_compressor_plugin {
  public:
  sz_plugin() {
    std::stringstream ss;
    ss << sz_plugin::major_version() << "." << sz_plugin::minor_version() << "." << sz_plugin::patch_version() << "." << revision_version();
    sz_version = ss.str();
    SZ_Init(NULL);
  };
  ~sz_plugin() {
    SZ_Finalize();
  }


  struct pressio_options get_configuration_impl() const override {
    struct pressio_options options;
    pressio_options_set_integer(&options, "pressio:thread_safe", pressio_thread_safety_serialized);
    return options;
  }

  struct pressio_options get_options_impl() const override {
    struct pressio_options options;
    options.set_type("sz:config_file", pressio_option_charptr_type);
    options.set_type("sz:config_struct", pressio_option_userptr_type);
    options.set("sz:max_quant_intervals", confparams_cpr->max_quant_intervals);
    options.set("sz:quantization_intervals ", confparams_cpr->quantization_intervals);
    options.set("sz:max_range_radius", confparams_cpr->maxRangeRadius);
    options.set("sz:sol_id", confparams_cpr->sol_ID);
    options.set("sz:lossless_compressor", confparams_cpr->losslessCompressor);
    options.set("sz:sample_distance", confparams_cpr->sampleDistance);
    options.set("sz:pred_threshold", confparams_cpr->predThreshold);
    options.set("sz:sz_mode", confparams_cpr->szMode);
    options.set("sz:gzip_mode", confparams_cpr->gzipMode);
    options.set("sz:error_bound_mode", confparams_cpr->errorBoundMode);
	  options.set("sz:abs_err_bound", confparams_cpr->absErrBound);
	  options.set("sz:rel_err_bound", confparams_cpr->relBoundRatio);
	  options.set("sz:psnr_err_bound", confparams_cpr->psnr);
	  options.set("sz:pw_rel_err_bound", confparams_cpr->pw_relBoundRatio);
	  options.set("sz:segment_size", confparams_cpr->segment_size);
	  options.set("sz:pwr_type", confparams_cpr->pwr_type);
	  options.set("sz:snapshot_cmpr_step", confparams_cpr->snapshotCmprStep);
	  options.set("sz:accelerate_pw_rel_compression", confparams_cpr->accelerate_pw_rel_compression);
	  options.set_type("sz:prediction_mode", pressio_option_int32_type);
	  options.set_type("sz:plus_bits", pressio_option_int32_type);
	  options.set_type("sz:random_access", pressio_option_int32_type);
    options.set_type("sz:data_type", pressio_option_double_type);
    options.set("sz:app", app.c_str());
    options.set("sz:user_params", user_params);
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override {

    struct sz_params* sz_param;
    std::string config_file;
    if(options.get("sz:config_file", &config_file) == pressio_options_key_set) {
      SZ_Finalize();
      SZ_Init(config_file.c_str());
    } else if (options.get("sz:config_struct", (void**)&sz_param) == pressio_options_key_set) {
      SZ_Finalize();
      SZ_Init_Params(sz_param);
    }

    options.get("sz:max_quant_intervals", &confparams_cpr->max_quant_intervals);
    options.get("sz:quantization_intervals ", &confparams_cpr->quantization_intervals);
    options.get("sz:max_range_radius", &confparams_cpr->maxRangeRadius);
    options.get("sz:sol_id", &confparams_cpr->sol_ID);
    options.get("sz:lossless_compressor", &confparams_cpr->losslessCompressor);
    options.get("sz:sample_distance", &confparams_cpr->sampleDistance);
    options.get("sz:pred_threshold", &confparams_cpr->predThreshold);
    options.get("sz:sz_mode", &confparams_cpr->szMode);
    options.get("sz:gzip_mode", &confparams_cpr->gzipMode);
    options.get("sz:error_bound_mode", &confparams_cpr->errorBoundMode);
    options.get("sz:abs_err_bound", &confparams_cpr->absErrBound);
    options.get("sz:rel_err_bound", &confparams_cpr->relBoundRatio);
    options.get("sz:psnr_err_bound", &confparams_cpr->psnr);
    options.get("sz:pw_rel_err_bound", &confparams_cpr->pw_relBoundRatio);
    options.get("sz:segment_size", &confparams_cpr->segment_size);
    options.get("sz:pwr_type", &confparams_cpr->pwr_type);
    options.get("sz:snapshot_cmpr_step", &confparams_cpr->snapshotCmprStep);
    options.get("sz:prediction_mode", &confparams_cpr->predictionMode);
    options.get("sz:accelerate_pw_rel_compression", &confparams_cpr->accelerate_pw_rel_compression);
    options.get("sz:plus_bits", &confparams_cpr->plus_bits);
    options.get("sz:random_access", &confparams_cpr->randomAccess);
    options.get("sz:data_type", &confparams_cpr->dataType);
    options.get("sz:app", &app);
    options.get("sz:user_params", &user_params);

    return 0;
  }

  int compress_impl(const pressio_data *input, struct pressio_data* output) override {
    size_t r1 = pressio_data_get_dimension(input, 0);
    size_t r2 = pressio_data_get_dimension(input, 1);
    size_t r3 = pressio_data_get_dimension(input, 2);
    size_t r4 = pressio_data_get_dimension(input, 3);
    size_t r5 = pressio_data_get_dimension(input, 4);
    int status = SZ_NSCS;
    size_t outsize = 0;
    unsigned char* compressed_data = SZ_compress_customize(app.c_str(), user_params,
        libpressio_type_to_sz_type(pressio_data_dtype(input)),
        pressio_data_ptr(input, nullptr),
        r5,
        r4,
        r3,
        r2,
        r1,
        &outsize,
        &status
        );
    *output = pressio_data::move(pressio_byte_dtype, compressed_data, 1, &outsize, pressio_data_libc_free_fn, nullptr);
    return 0;
  }
  int decompress_impl(const pressio_data *input, struct pressio_data* output) override {

    size_t r[] = {
     pressio_data_get_dimension(output, 0),
     pressio_data_get_dimension(output, 1),
     pressio_data_get_dimension(output, 2),
     pressio_data_get_dimension(output, 3),
     pressio_data_get_dimension(output, 4),
    };
    size_t ndims = pressio_data_num_dimensions(output);

    int status = SZ_NSCS;
    pressio_dtype type = pressio_data_dtype(output);
    void* decompressed_data = SZ_decompress_customize(
        app.c_str(),
        user_params,
        libpressio_type_to_sz_type(type),
        (unsigned char*)pressio_data_ptr(input, nullptr),
        pressio_data_get_dimension(input, 0),
        r[4],
        r[3],
        r[2],
        r[1],
        r[0],
        &status
        );
    *output = pressio_data::move(type, decompressed_data, ndims, r, pressio_data_libc_free_fn, nullptr);
    return 0;
  }

  int major_version() const override {
    return SZ_VER_MAJOR;
  }
  int minor_version() const override {
    return SZ_VER_MINOR;
  }
  int patch_version() const override {
    return SZ_VER_BUILD;
  }
  int revision_version () const { 
    return SZ_VER_REVISION;
  }

  const char* version() const override {
    return sz_version.c_str(); 
  }

  const char* prefix() const override {
    return "sz";
  }


  private:
  static int libpressio_type_to_sz_type(pressio_dtype type) {
    switch(type)
    {
      case pressio_float_dtype:  return SZ_FLOAT;
      case pressio_double_dtype: return SZ_DOUBLE;
      case pressio_uint8_dtype: return SZ_UINT8;
      case pressio_int8_dtype: return SZ_INT8;
      case pressio_uint16_dtype: return SZ_UINT16;
      case pressio_int16_dtype: return SZ_INT16;
      case pressio_uint32_dtype: return SZ_UINT32;
      case pressio_int32_dtype: return SZ_INT32;
      case pressio_uint64_dtype: return SZ_UINT64;
      case pressio_int64_dtype: return SZ_INT64;
      case pressio_byte_dtype: return SZ_INT8;
    }
    return -1;
  }
  std::string sz_version;
  std::string app = "SZ";
  void* user_params = nullptr;
};

std::unique_ptr<libpressio_compressor_plugin> make_c_sz() {
  return compat::make_unique<sz_plugin>();
}

static pressio_register X(compressor_plugins(), "sz", [](){ static auto sz = std::make_shared<sz_plugin>(); return sz; });
