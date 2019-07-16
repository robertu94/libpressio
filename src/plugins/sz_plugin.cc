#include <memory>
#include <sstream>

#include <sz.h>

#include "liblossy_plugin.h"
#include "lossy_data.h"
#include "lossy_options.h"
#include "lossy_option.h"


class sz_plugin: public liblossy_plugin {
  public:
  sz_plugin() {
    std::stringstream ss;
    ss << major_version() << "." << minor_version() << "." << patch_version() << "." << revision_version();
    sz_version = ss.str();
    SZ_Init(NULL);
  };
  ~sz_plugin() {
    SZ_Finalize();
  }

  virtual struct lossy_options* get_options() const override {
    struct lossy_options* options = lossy_options_new();
    lossy_options_clear(options, "sz:config_file");
    lossy_options_clear(options, "sz:config_struct");
    lossy_options_set_integer(options, "sz:mode", confparams_cpr->szMode);
    lossy_options_set_uinteger(options, "sz:max_quant_intervals", confparams_cpr->max_quant_intervals);
    lossy_options_set_uinteger(options, "sz:quantization_intervals ", confparams_cpr->quantization_intervals);
    lossy_options_set_uinteger(options, "sz:max_range_radius", confparams_cpr->maxRangeRadius);
    lossy_options_set_integer(options, "sz:sol_id", confparams_cpr->sol_ID);
    lossy_options_set_integer(options, "sz:lossless_compressor", confparams_cpr->losslessCompressor);
    lossy_options_set_integer(options, "sz:sample_distance", confparams_cpr->sampleDistance);
    lossy_options_set_float(options, "sz:pred_threshold", confparams_cpr->predThreshold);
    lossy_options_set_integer(options, "sz:sz_mode", confparams_cpr->szMode);
    lossy_options_set_integer(options, "sz:gzip_mode", confparams_cpr->gzipMode);
    lossy_options_set_integer(options, "sz:error_bound_mode", confparams_cpr->errorBoundMode);
	  lossy_options_set_double(options, "sz:abs_err_bound", confparams_cpr->absErrBound);
	  lossy_options_set_double(options, "sz:rel_err_bound", confparams_cpr->relBoundRatio);
	  lossy_options_set_double(options, "sz:psnr_err_bound", confparams_cpr->psnr);
	  lossy_options_set_double(options, "sz:rel_err_bound", confparams_cpr->pw_relBoundRatio);
	  lossy_options_set_integer(options, "sz:segment_size", confparams_cpr->segment_size);
	  lossy_options_set_integer(options, "sz:pwr_type", confparams_cpr->pwr_type);
	  lossy_options_set_integer(options, "sz:snapshot_cmpr_step", confparams_cpr->snapshotCmprStep);
	  lossy_options_set_integer(options, "sz:accelerate_pw_rel_compression", confparams_cpr->accelerate_pw_rel_compression);
	  lossy_options_clear(options, "sz:prediction_mode");
	  lossy_options_clear(options, "sz:plus_bits");
	  lossy_options_clear(options, "sz:random_access");
    lossy_options_clear(options, "sz:data_type");
    return options;
  }

  virtual int set_options(struct lossy_options const* options) override {

    struct sz_params* sz_param;
    const char* config_file;
    if(lossy_options_get_string(options, "sz:config_file", &config_file) == lossy_options_key_set) {
      SZ_Finalize();
      SZ_Init(config_file);
    } else if (lossy_options_get_userptr(options, "sz:config_struct", (void**)&sz_param) == lossy_options_key_set) {
      SZ_Finalize();
      SZ_Init_Params(sz_param);
    }

    lossy_options_get_integer(options, "sz:mode", &confparams_cpr->szMode);
    lossy_options_get_uinteger(options, "sz:max_quant_intervals", &confparams_cpr->max_quant_intervals);
    lossy_options_get_uinteger(options, "sz:quantization_intervals ", &confparams_cpr->quantization_intervals);
    lossy_options_get_uinteger(options, "sz:max_range_radius", &confparams_cpr->maxRangeRadius);
    lossy_options_get_integer(options, "sz:sol_id", &confparams_cpr->sol_ID);
    lossy_options_get_integer(options, "sz:lossless_compressor", &confparams_cpr->losslessCompressor);
    lossy_options_get_integer(options, "sz:sample_distance", &confparams_cpr->sampleDistance);
    lossy_options_get_float(options, "sz:pred_threshold", &confparams_cpr->predThreshold);
    lossy_options_get_integer(options, "sz:sz_mode", &confparams_cpr->szMode);
    lossy_options_get_integer(options, "sz:gzip_mode", &confparams_cpr->gzipMode);
    lossy_options_get_integer(options, "sz:error_bound_mode", &confparams_cpr->errorBoundMode);
    lossy_options_get_double(options, "sz:abs_err_bound", &confparams_cpr->absErrBound);
    lossy_options_get_double(options, "sz:rel_err_bound", &confparams_cpr->relBoundRatio);
    lossy_options_get_double(options, "sz:psnr_err_bound", &confparams_cpr->psnr);
    lossy_options_get_double(options, "sz:rel_err_bound", &confparams_cpr->pw_relBoundRatio);
    lossy_options_get_integer(options, "sz:segment_size", &confparams_cpr->segment_size);
    lossy_options_get_integer(options, "sz:pwr_type", &confparams_cpr->pwr_type);
    lossy_options_get_integer(options, "sz:snapshot_cmpr_step", &confparams_cpr->snapshotCmprStep);
    lossy_options_get_integer(options, "sz:prediction_mode", &confparams_cpr->predictionMode);
    lossy_options_get_integer(options, "sz:accelerate_pw_rel_compression", &confparams_cpr->accelerate_pw_rel_compression);
    lossy_options_get_integer(options, "sz:plus_bits", &confparams_cpr->plus_bits);
    lossy_options_get_integer(options, "sz:random_access", &confparams_cpr->randomAccess);
    lossy_options_get_integer(options, "sz:data_type", &confparams_cpr->dataType);

    return 0;
  }

  int compress(struct lossy_data* input, struct lossy_data** output) override {
    size_t r1 = lossy_data_get_dimention(input, 0);
    size_t r2 = lossy_data_get_dimention(input, 1);
    size_t r3 = lossy_data_get_dimention(input, 2);
    size_t r4 = lossy_data_get_dimention(input, 3);
    size_t r5 = lossy_data_get_dimention(input, 4);
    size_t outsize = 0;
    lossy_data_free(*output);
    unsigned char* compressed_data = SZ_compress(
        liblossy_type_to_sz_type(lossy_data_dtype(input)),
        lossy_data_ptr(input, nullptr),
        &outsize,
        r5,
        r4,
        r3,
        r2,
        r1);
    *output = lossy_data_new(lossy_byte_dtype, compressed_data, 1, &outsize);
    return 0;
  }
  int decompress(struct lossy_data* input, struct lossy_data** output) override {

    size_t r[] = {
     lossy_data_get_dimention(*output, 0),
     lossy_data_get_dimention(*output, 1),
     lossy_data_get_dimention(*output, 2),
     lossy_data_get_dimention(*output, 3),
     lossy_data_get_dimention(*output, 4),
    };
    size_t ndims = lossy_data_num_dimentions(*output);

    lossy_dtype type = lossy_data_dtype(*output);
    void* decompressed_data = SZ_decompress(
        liblossy_type_to_sz_type(type),
        (unsigned char*)lossy_data_ptr(input, nullptr),
        lossy_data_get_dimention(input, 0),
        r[4],
        r[3],
        r[2],
        r[1],
        r[0]
        );
    lossy_data_free(*output);
    *output = lossy_data_new(type, decompressed_data, ndims, r);
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
  static int liblossy_type_to_sz_type(lossy_dtype type) {
    switch(type)
    {
      case lossy_float_dtype:  return SZ_FLOAT;
      case lossy_double_dtype: return SZ_DOUBLE;
      case lossy_uint8_dtype: return SZ_UINT8;
      case lossy_int8_dtype: return SZ_INT8;
      case lossy_uint16_dtype: return SZ_UINT16;
      case lossy_int16_dtype: return SZ_INT16;
      case lossy_uint32_dtype: return SZ_UINT32;
      case lossy_int32_dtype: return SZ_INT32;
      case lossy_uint64_dtype: return SZ_UINT64;
      case lossy_int64_dtype: return SZ_INT64;
      case lossy_byte_dtype: return SZ_INT8;
    }
    return -1;
  }
  std::string sz_version;
};

std::unique_ptr<liblossy_plugin> make_sz() {
  return std::make_unique<sz_plugin>();
}
