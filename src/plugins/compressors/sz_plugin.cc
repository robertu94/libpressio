#include <algorithm>
#include <iterator>
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
#include "libpressio_ext/compat/memory.h"

#define PRESSIO_SZ_VERSION_GREATEREQ(major, minor, build, revision) \
   (SZ_VER_MAJOR > major || \
   (SZ_VER_MAJOR == major && SZ_VER_MINOR > minor) ||                                  \
   (SZ_VER_MAJOR == major && SZ_VER_MINOR == minor && SZ_VER_BUILD > build) || \
   (SZ_VER_MAJOR == major && SZ_VER_MINOR == minor && SZ_VER_BUILD == build && SZ_VER_REVISION >= revision))

namespace {
  struct iless {
    bool operator()(std::string lhs, std::string rhs) const {
      std::transform(std::begin(lhs), std::end(lhs), std::begin(lhs), [](unsigned char c){return std::tolower(c);});
      std::transform(std::begin(rhs), std::end(rhs), std::begin(rhs), [](unsigned char c){return std::tolower(c);});
      return lhs < rhs;
    }
  };
  static std::map<std::string, int, iless> const sz_mode_str_to_code {
    {"abs", ABS},
    {"rel", REL},
    {"vr_rel", REL},
    {"abs_and_rel", ABS_AND_REL},
    {"abs_or_rel", ABS_OR_REL},
    {"psnr", PSNR},
#if PRESSIO_SZ_VERSION_GREATEREQ(2,1,8,3)
    {"norm", NORM},
#endif
    {"pw_rel", PW_REL},
    {"abs_or_pw_rel", ABS_OR_PW_REL},
    {"abs_and_pw_rel", ABS_AND_PW_REL},
    {"rel_or_pw_rel", REL_OR_PW_REL},
    {"rel_and_pw_rel", REL_AND_PW_REL},
  };
}

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
    set(options, "pressio:thread_safe", (unsigned int)pressio_thread_safety_serialized);

    std::vector<std::string> vs;
    std::transform(
        std::begin(sz_mode_str_to_code),
        std::end(sz_mode_str_to_code),
        std::back_inserter(vs),
        [](typename decltype(sz_mode_str_to_code)::const_reference m){ return m.first; });
    set(options, "sz:error_bound_mode_str", vs);
    return options;
  }

  struct pressio_options get_options_impl() const override {
    struct pressio_options options;
    set_type(options, "sz:config_file", pressio_option_charptr_type);
    set_type(options, "sz:config_struct", pressio_option_userptr_type);
#if PRESSIO_SZ_VERSION_GREATEREQ(2,1,9,0)
    set(options, "sz:protect_value_range", confparams_cpr->protectValueRange);
#endif
    set(options, "sz:max_quant_intervals", confparams_cpr->max_quant_intervals);
    set(options, "sz:quantization_intervals", confparams_cpr->quantization_intervals);
    set(options, "sz:sol_id", confparams_cpr->sol_ID);
    set(options, "sz:lossless_compressor", confparams_cpr->losslessCompressor);
    set(options, "sz:sample_distance", confparams_cpr->sampleDistance);
    set(options, "sz:pred_threshold", confparams_cpr->predThreshold);
    set(options, "sz:sz_mode", confparams_cpr->szMode);
    set(options, "sz:gzip_mode", confparams_cpr->gzipMode);
    set(options, "sz:error_bound_mode", confparams_cpr->errorBoundMode);
    set_type(options, "sz:error_bound_mode_str", pressio_option_charptr_type);
	  set(options, "sz:abs_err_bound", confparams_cpr->absErrBound);
	  set(options, "sz:rel_err_bound", confparams_cpr->relBoundRatio);
	  set(options, "sz:psnr_err_bound", confparams_cpr->psnr);
	  set(options, "sz:pw_rel_err_bound", confparams_cpr->pw_relBoundRatio);
	  set(options, "sz:segment_size", confparams_cpr->segment_size);
	  set(options, "sz:snapshot_cmpr_step", confparams_cpr->snapshotCmprStep);
	  set(options, "sz:accelerate_pw_rel_compression", confparams_cpr->accelerate_pw_rel_compression);
	  set_type(options, "sz:prediction_mode", pressio_option_int32_type);
    set_type(options, "sz:data_type", pressio_option_double_type);
    set(options, "sz:app", app.c_str());
    set(options, "sz:user_params", user_params);
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override {

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
      auto key = sz_mode_str_to_code.find(error_bound_mode_str);
      if(key != sz_mode_str_to_code.end()) {
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
    get(options, "sz:app", &app);
    get(options, "sz:user_params", &user_params);

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

  std::shared_ptr<libpressio_compressor_plugin> clone() override {
    return compressor_plugins().build("sz");
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

static pressio_register compressor_sz_plugin(compressor_plugins(), "sz", [](){ static auto sz = std::make_shared<sz_plugin>(); return sz; });
