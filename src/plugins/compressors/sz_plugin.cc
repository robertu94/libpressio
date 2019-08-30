#include <memory>
#include <sstream>
#include <cstdlib>

#include <sz/sz.h>

#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "pressio_data.h"
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

  struct pressio_options* get_options_impl() const override {
    struct pressio_options* options = pressio_options_new();
    pressio_options_set_type(options, "sz:config_file", pressio_option_charptr_type);
    pressio_options_set_type(options, "sz:config_struct", pressio_option_userptr_type);
    pressio_options_set_uinteger(options, "sz:max_quant_intervals", confparams_cpr->max_quant_intervals);
    pressio_options_set_uinteger(options, "sz:quantization_intervals ", confparams_cpr->quantization_intervals);
    pressio_options_set_uinteger(options, "sz:max_range_radius", confparams_cpr->maxRangeRadius);
    pressio_options_set_integer(options, "sz:sol_id", confparams_cpr->sol_ID);
    pressio_options_set_integer(options, "sz:lossless_compressor", confparams_cpr->losslessCompressor);
    pressio_options_set_integer(options, "sz:sample_distance", confparams_cpr->sampleDistance);
    pressio_options_set_float(options, "sz:pred_threshold", confparams_cpr->predThreshold);
    pressio_options_set_integer(options, "sz:sz_mode", confparams_cpr->szMode);
    pressio_options_set_integer(options, "sz:gzip_mode", confparams_cpr->gzipMode);
    pressio_options_set_integer(options, "sz:error_bound_mode", confparams_cpr->errorBoundMode);
	  pressio_options_set_double(options, "sz:abs_err_bound", confparams_cpr->absErrBound);
	  pressio_options_set_double(options, "sz:rel_err_bound", confparams_cpr->relBoundRatio);
	  pressio_options_set_double(options, "sz:psnr_err_bound", confparams_cpr->psnr);
	  pressio_options_set_double(options, "sz:pw_rel_err_bound", confparams_cpr->pw_relBoundRatio);
	  pressio_options_set_integer(options, "sz:segment_size", confparams_cpr->segment_size);
	  pressio_options_set_integer(options, "sz:pwr_type", confparams_cpr->pwr_type);
	  pressio_options_set_integer(options, "sz:snapshot_cmpr_step", confparams_cpr->snapshotCmprStep);
	  pressio_options_set_integer(options, "sz:accelerate_pw_rel_compression", confparams_cpr->accelerate_pw_rel_compression);
	  pressio_options_set_type(options, "sz:prediction_mode", pressio_option_int32_type);
	  pressio_options_set_type(options, "sz:plus_bits", pressio_option_int32_type);
	  pressio_options_set_type(options, "sz:random_access", pressio_option_int32_type);
    pressio_options_set_type(options, "sz:data_type", pressio_option_double_type);
    pressio_options_set_string(options, "sz:app", app.c_str());
    pressio_options_set_userptr(options, "sz:user_params", user_params);
    return options;
  }

  int set_options_impl(struct pressio_options const* options) override {

    struct sz_params* sz_param;
    const char* config_file;
    if(pressio_options_get_string(options, "sz:config_file", &config_file) == pressio_options_key_set) {
      SZ_Finalize();
      SZ_Init(config_file);
    } else if (pressio_options_get_userptr(options, "sz:config_struct", (void**)&sz_param) == pressio_options_key_set) {
      SZ_Finalize();
      SZ_Init_Params(sz_param);
    }

    pressio_options_get_uinteger(options, "sz:max_quant_intervals", &confparams_cpr->max_quant_intervals);
    pressio_options_get_uinteger(options, "sz:quantization_intervals ", &confparams_cpr->quantization_intervals);
    pressio_options_get_uinteger(options, "sz:max_range_radius", &confparams_cpr->maxRangeRadius);
    pressio_options_get_integer(options, "sz:sol_id", &confparams_cpr->sol_ID);
    pressio_options_get_integer(options, "sz:lossless_compressor", &confparams_cpr->losslessCompressor);
    pressio_options_get_integer(options, "sz:sample_distance", &confparams_cpr->sampleDistance);
    pressio_options_get_float(options, "sz:pred_threshold", &confparams_cpr->predThreshold);
    pressio_options_get_integer(options, "sz:sz_mode", &confparams_cpr->szMode);
    pressio_options_get_integer(options, "sz:gzip_mode", &confparams_cpr->gzipMode);
    pressio_options_get_integer(options, "sz:error_bound_mode", &confparams_cpr->errorBoundMode);
    pressio_options_get_double(options, "sz:abs_err_bound", &confparams_cpr->absErrBound);
    pressio_options_get_double(options, "sz:rel_err_bound", &confparams_cpr->relBoundRatio);
    pressio_options_get_double(options, "sz:psnr_err_bound", &confparams_cpr->psnr);
    pressio_options_get_double(options, "sz:pw_rel_err_bound", &confparams_cpr->pw_relBoundRatio);
    pressio_options_get_integer(options, "sz:segment_size", &confparams_cpr->segment_size);
    pressio_options_get_integer(options, "sz:pwr_type", &confparams_cpr->pwr_type);
    pressio_options_get_integer(options, "sz:snapshot_cmpr_step", &confparams_cpr->snapshotCmprStep);
    pressio_options_get_integer(options, "sz:prediction_mode", &confparams_cpr->predictionMode);
    pressio_options_get_integer(options, "sz:accelerate_pw_rel_compression", &confparams_cpr->accelerate_pw_rel_compression);
    pressio_options_get_integer(options, "sz:plus_bits", &confparams_cpr->plus_bits);
    pressio_options_get_integer(options, "sz:random_access", &confparams_cpr->randomAccess);
    pressio_options_get_integer(options, "sz:data_type", &confparams_cpr->dataType);
    const char* tmp_app;
    if(pressio_options_get_string(options, "sz:app", &tmp_app) == pressio_options_key_set)
    {
      app = tmp_app;
      free((void*)tmp_app);
    }
    pressio_options_get_userptr(options, "sz:user_params", &user_params);

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
  return std::make_unique<sz_plugin>();
}
