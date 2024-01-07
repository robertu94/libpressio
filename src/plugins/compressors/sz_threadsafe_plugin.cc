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
#include "std_compat/std_compat.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "pressio_option.h"
#include "std_compat/memory.h"
#include "sz_common.h"
#include "iless.h"

namespace libpressio { namespace sz_threadsafe {

std::string get_version_sz_threadsafe(){
	static std::string
		s=[]{
			std::stringstream ss;
			ss << SZ_VER_MAJOR << "." << SZ_VER_MINOR << "." << SZ_VER_BUILD << "." << SZ_VER_REVISION;
			return ss.str();
		}();
	return s.c_str();
}
static std::map<std::string, int, iless> const sz_threadsafe_mode_str_to_code {
    {"abs", ABS},
    {"rel", REL},
    {"vr_rel", REL},
    {"abs_and_rel", ABS_AND_REL},
    {"abs_or_rel", ABS_OR_REL},
    {"pw_rel", PW_REL},
    {"abs_or_pw_rel", ABS_OR_PW_REL},
    {"abs_and_pw_rel", ABS_AND_PW_REL},
    {"rel_or_pw_rel", REL_OR_PW_REL},
    {"rel_and_pw_rel", REL_AND_PW_REL},
};
class sz_threadsafe_plugin: public libpressio_compressor_plugin {
  public:
  sz_threadsafe_plugin(std::shared_ptr<sz_init_handle>&& init_handle): init_handle(init_handle) {
    compat::shared_lock<compat::shared_mutex> lock(init_handle->sz_init_lock);
    threadsafe_params = *confparams_cpr;
  };
  ~sz_threadsafe_plugin() {
  }


  struct pressio_options get_configuration_impl() const override {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "stable");
    set(options, "pressio:lossless:min", 0);
    set(options, "pressio:lossless:max", 0);
#ifdef HAVE_RANDOMACCESS
    set(options, "sz_threadsafe:random_access_enabled", 1u);
#else
    set(options, "sz_threadsafe:random_access_enabled", 0u);
#endif
#ifdef HAVE_TIMECMPR
    set(options, "sz_threadsafe:timecmpr", 1u);
#else
    set(options, "sz_threadsafe:timecmpr", 0u);
#endif
#ifdef HAVE_PASTRI
    set(options, "sz_threadsafe:pastri", 1u);
#else
    set(options, "sz_threadsafe:pastri", 0u);
#endif


    std::vector<std::string> vs;
    std::transform(
        std::begin(sz_threadsafe_mode_str_to_code),
        std::end(sz_threadsafe_mode_str_to_code),
        std::back_inserter(vs),
        [](typename decltype(sz_threadsafe_mode_str_to_code)::const_reference m){ return m.first; });
    set(options, "sz_threadsafe:error_bound_mode_str", vs);
    
        std::vector<std::string> invalidations {"sz_threadsafe:protect_value_range", "sz_threadsafe:max_quant_intervals", "sz_threadsafe:quantization_intervals", "sz_threadsafe:sol_id", "sz_threadsafe:lossless_compressor", "sz_threadsafe:sample_distance", "sz_threadsafe:pred_threshold", "sz_threadsafe:sz_mode", "sz_threadsafe:gzip_mode", "sz_threadsafe:abs_err_bound", "sz_threadsafe:rel_err_bound", "sz_threadsafe:psnr_err_bound", "sz_threadsafe:pw_rel_err_bound", "sz_threadsafe:segment_size", "sz_threadsafe:snapshot_cmpr_step", "sz_threadsafe:prediction_mode", "sz_threadsafe:accelerate_pw_rel_compression", "sz_threadsafe:data_type", "sz_threadsafe:app", "sz_threadsafe:user_params", "pressio:abs", "pressio:rel", "pressio:lossless", "sz_threadsafe:config_struct", "sz_threadsafe:error_bound_mode_str", "sz_threadsafe:error_bound_mode", "sz_threadsafe:exafel:tolerance", "sz_threadsafe:exafel:peaks_segs", "sz_threadsafe:exafel:peaks_rows", "sz_threadsafe:exafel:peaks_cols", "sz_threadsafe:exafel:num_peaks", "sz_threadsafe:exafel:calib_panel", "sz_threadsafe:exafel:bin_size", "sz_threadsafe:exafel:sz_dim", "sz_threadsafe:exafel:peak_size"}; 
        std::vector<pressio_configurable const*> invalidation_children {}; 
        
        set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
        set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
        set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));

    
        set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{"pressio:abs", "pressio:rel", "pressio:lossless"}));

    return options;
  }

  struct pressio_options get_documentation_impl() const override {
    struct pressio_options options;
    set(options, "pressio:description", R"(SZ threadsafe is a threadsafe version of the error bounded lossy compressor SZ that uses prediction 
      based methods to compress data. SZ threadsafe has fewer features than SZ, including but not limited to: no write stats support,
      and no PSNR/NORM mode. More information can be found about SZ on its [project homepage](https://github.com/disheng222/sz).)");
    set(options, "sz_threadsafe:random_access_enabled", "true if SZ was compiled in random access mode");
    set(options, "sz_threadsafe:timecmpr", "true if SZ if SZ is compiled in time based compression mode");
    set(options, "sz_threadsafe:pastri", "true if PASTRI mode was built");
    set(options, "sz_threadsafe:write_stats", "true if SZ is compiled with compression statistics support");


    set(options, "sz_threadsafe:abs_err_bound", "the absolute error bound ");
    set(options, "sz_threadsafe:accelerate_pw_rel_compression", "trade compression ratio for a faster pw_rel compression");
    set(options, "sz_threadsafe:app", "access a application specific mode of SZ");
    set(options, "sz_threadsafe:config_struct", "structure passed to SZ_Init_Params()" );
    set(options, "sz_threadsafe:data_type", "an internal option to control compression");
    set(options, "sz_threadsafe:error_bound_mode", "integer code used to determine error bound mode");
    set(options, "sz_threadsafe:error_bound_mode_str", "human readable string to set the error bound mode");
    set(options, "sz_threadsafe:exafel:bin_size", "for ROIBIN-SZ, the size of the binning applied");
    set(options, "sz_threadsafe:exafel:calib_panel", "for ROIBIN-SZ the size of the calibration panel");
    set(options, "sz_threadsafe:exafel:num_peaks", "for ROIBIN-SZ the number of peaks");
    set(options, "sz_threadsafe:exafel:peak_size", "for ROIBIN-SZ the size of the peaks");
    set(options, "sz_threadsafe:exafel:peaks_cols", "for ROIBIN-SZ the list of columns peaks appear in");
    set(options, "sz_threadsafe:exafel:peaks_rows", "for ROIBIN-SZ the list of rows peaks appear in");
    set(options, "sz_threadsafe:exafel:peaks_segs", "for ROIBIN-SZ the segments peaks appear in");
    set(options, "sz_threadsafe:exafel:sz_dim", "for ROIBIN-SZ the SZ dimensionality prefered");
    set(options, "sz_threadsafe:exafel:tolerance", "for ROIBIN-SZ the tolerance used after binning");
    set(options, "sz_threadsafe:gzip_mode", "Which mode to pass to GZIP when used");
    set(options, "sz_threadsafe:lossless_compressor", "Which lossless compressor to use for stage 4");
    set(options, "sz_threadsafe:max_quant_intervals", "the maximum number of quantization intervals");
    set(options, "sz_threadsafe:pred_threshold", "an internal option used to control compression");
    set(options, "sz_threadsafe:prediction_mode", "an internal option used to control compression");
    set(options, "sz_threadsafe:psnr_err_bound", "the bound on the error in the PSNR");
    set(options, "sz_threadsafe:pw_rel_err_bound", "the bound on the pointwise relative error");
    set(options, "sz_threadsafe:quantization_intervals", "the number of quantization intervals to use, 0 means automatic");
    set(options, "sz_threadsafe:random_access", "internal options to use random access mode when compiled in");
    set(options, "sz_threadsafe:rel_err_bound", "the value range relative error bound mode");
    set(options, "sz_threadsafe:sample_distance", "internal option used to control compression");
    set(options, "sz_threadsafe:segment_size", "internal option used to control compression. number of points in each segement for pw_relBoundRatio");
    set(options, "sz_threadsafe:snapshot_cmpr_step", "the frequency of preforming single snapshot based compression in time based compression");
    set(options, "sz_threadsafe:sol_id", "an internal option use d to control compression");
    set(options, "sz_threadsafe:sz_mode", "SZ Mode either SZ_BEST_COMPRESSION or SZ_BEST_SPEED");
    set(options, "sz_threadsafe:user_params", "arguments passed to the application specific mode of SZ. Use in conjunction with sz:app");
    set(options, "sz_threadsafe:protect_value_range", "should the value range be preserved during compression");
    return options;
  }

  sz_params threadsafe_params;


  struct pressio_options get_options_impl() const override {
    struct pressio_options options;
    set_type(options, "sz_threadsafe:config_struct", pressio_option_userptr_type);
#if PRESSIO_SZ_VERSION_GREATEREQ(2,1,9,0)
    set(options, "sz_threadsafe:protect_value_range", threadsafe_params.protectValueRange);
#endif
    set(options, "sz_threadsafe:max_quant_intervals", threadsafe_params.max_quant_intervals);
    set(options, "sz_threadsafe:quantization_intervals", threadsafe_params.quantization_intervals);
    set(options, "sz_threadsafe:sol_id", threadsafe_params.sol_ID);
    set(options, "sz_threadsafe:lossless_compressor", threadsafe_params.losslessCompressor);
    set(options, "sz_threadsafe:sample_distance", threadsafe_params.sampleDistance);
    set(options, "sz_threadsafe:pred_threshold", threadsafe_params.predThreshold);
    set(options, "sz_threadsafe:sz_mode", threadsafe_params.szMode);
    set(options, "sz_threadsafe:gzip_mode", threadsafe_params.gzipMode);
    set(options, "sz_threadsafe:error_bound_mode", threadsafe_params.errorBoundMode);
    set_type(options, "sz_threadsafe:error_bound_mode_str", pressio_option_charptr_type);
	  set(options, "sz_threadsafe:abs_err_bound", threadsafe_params.absErrBound);
	  set(options, "sz_threadsafe:rel_err_bound", threadsafe_params.relBoundRatio);
	  set(options, "sz_threadsafe:psnr_err_bound", threadsafe_params.psnr);
	  set(options, "sz_threadsafe:pw_rel_err_bound", threadsafe_params.pw_relBoundRatio);
	  set(options, "sz_threadsafe:segment_size", threadsafe_params.segment_size);
	  set(options, "sz_threadsafe:snapshot_cmpr_step", threadsafe_params.snapshotCmprStep);
	  set(options, "sz_threadsafe:accelerate_pw_rel_compression", threadsafe_params.accelerate_pw_rel_compression);
	  set_type(options, "sz_threadsafe:prediction_mode", pressio_option_int32_type);
    set_type(options, "sz_threadsafe:data_type", pressio_option_double_type);
#if PRESSIO_SZ_VERSION_GREATEREQ(2,1,11,1)
    set(options, "sz_threadsafe:exafel:peaks_segs", exafel_peaks_segs);
    set(options, "sz_threadsafe:exafel:peaks_rows", exafel_peaks_rows);
    set(options, "sz_threadsafe:exafel:peaks_cols", exafel_peaks_cols);
    set(options, "sz_threadsafe:exafel:num_peaks", static_cast<unsigned int>(exafel_params.numPeaks));
    set(options, "sz_threadsafe:exafel:calib_panel", exafel_calibPanel);
    set(options, "sz_threadsafe:exafel:tolerance", exafel_params.tolerance);
    set(options, "sz_threadsafe:exafel:bin_size", static_cast<unsigned int>(exafel_params.binSize));
    set(options, "sz_threadsafe:exafel:sz_dim", static_cast<unsigned int>(exafel_params.szDim));
    set(options, "sz_threadsafe:exafel:peak_size", static_cast<unsigned int>(exafel_params.peakSize));
#endif
    set(options, "sz_threadsafe:app", app.c_str());
#ifdef HAVE_RANDOMACCESS
    set(options, "sz_threadsafe:random_access", threadsafe_params.randomAccess);
#endif
    set(options, "sz_threadsafe:user_params", user_params);
    if(threadsafe_params.errorBoundMode == ABS) {
      set(options, "pressio:abs", threadsafe_params.absErrBound);
    } else {
      set_type(options, "pressio:abs", pressio_option_double_type);
    }
    if(threadsafe_params.errorBoundMode == REL) {
      set(options, "pressio:rel", threadsafe_params.relBoundRatio);
    } else {
      set_type(options, "pressio:rel", pressio_option_double_type);
    }
    if(threadsafe_params.errorBoundMode == ABS && threadsafe_params.absErrBound == 0) {
      set(options, "pressio:lossless", 0);
    } else {
      set_type(options, "pressio:lossless", pressio_option_int32_type);
    }
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override {
    if(get(options, "pressio:abs", &threadsafe_params.absErrBound) == pressio_options_key_set) {
      threadsafe_params.errorBoundMode = ABS;
    }
    if(get(options, "pressio:rel", &threadsafe_params.relBoundRatio) == pressio_options_key_set) {
      threadsafe_params.errorBoundMode = REL;
    }
    int32_t tmp = 0;
    if(get(options, "pressio:lossless", &tmp) == pressio_options_key_set) {
      threadsafe_params.errorBoundMode = ABS;
      threadsafe_params.absErrBound = 0.0;
    }

    struct sz_params* sz_threadsafe_param;
    if (get(options, "sz_threadsafe:config_struct",(void **) &sz_threadsafe_param) == pressio_options_key_set) {
	    memcpy(&threadsafe_params,sz_threadsafe_param,sizeof(sz_params));
    }

#if PRESSIO_SZ_VERSION_GREATEREQ(2,1,9,0)
    get(options, "sz_threadsafe:protect_value_range", &threadsafe_params.protectValueRange);
#endif
    get(options, "sz_threadsafe:max_quant_intervals", &threadsafe_params.max_quant_intervals);
    get(options, "sz_threadsafe:quantization_intervals", &threadsafe_params.quantization_intervals);
    get(options, "sz_threadsafe:sol_id", &threadsafe_params.sol_ID);
    get(options, "sz_threadsafe:lossless_compressor", &threadsafe_params.losslessCompressor);
    get(options, "sz_threadsafe:sample_distance", &threadsafe_params.sampleDistance);
    get(options, "sz_threadsafe:pred_threshold", &threadsafe_params.predThreshold);
    get(options, "sz_threadsafe:sz_mode", &threadsafe_params.szMode);
    get(options, "sz_threadsafe:gzip_mode", &threadsafe_params.gzipMode);

    std::string error_bound_mode_str;
    if(get(options, "sz_threadsafe:error_bound_mode_str", &error_bound_mode_str) == pressio_options_key_set) {
      auto key = sz_threadsafe_mode_str_to_code.find(error_bound_mode_str);
      if(key != sz_threadsafe_mode_str_to_code.end()) {
        threadsafe_params.errorBoundMode = key->second;
      }
    } else { 
      get(options, "sz_threadsafe:error_bound_mode", &threadsafe_params.errorBoundMode ); 
    }
    get(options, "sz_threadsafe:abs_err_bound", &threadsafe_params.absErrBound);
    get(options, "sz_threadsafe:rel_err_bound", &threadsafe_params.relBoundRatio);
    get(options, "sz_threadsafe:psnr_err_bound", &threadsafe_params.psnr);
    get(options, "sz_threadsafe:pw_rel_err_bound", &threadsafe_params.pw_relBoundRatio);
    get(options, "sz_threadsafe:segment_size", &threadsafe_params.segment_size);
    get(options, "sz_threadsafe:snapshot_cmpr_step", &threadsafe_params.snapshotCmprStep);
    get(options, "sz_threadsafe:prediction_mode", &threadsafe_params.predictionMode);
    get(options, "sz_threadsafe:accelerate_pw_rel_compression", &threadsafe_params.accelerate_pw_rel_compression);
    get(options, "sz_threadsafe:data_type", &threadsafe_params.dataType);
    get(options, "sz_threadsafe:app", &app);
    get(options, "sz_threadsafe:user_params", &user_params);
#if PRESSIO_SZ_VERSION_GREATEREQ(2,1,11,1)
{
    unsigned int temp;
    if(get(options, "sz_threadsafe:exafel:peaks_segs", &exafel_peaks_segs) == pressio_options_key_set) {
      exafel_params.peaksSegs = static_cast<uint16_t*>(exafel_peaks_segs.data());
    }
    if(get(options, "sz_threadsafe:exafel:peaks_rows", &exafel_peaks_rows) == pressio_options_key_set) {
      exafel_params.peaksRows = static_cast<uint16_t*>(exafel_peaks_rows.data());
    }
    if(get(options, "sz_threadsafe:exafel:peaks_cols", &exafel_peaks_cols) == pressio_options_key_set) {
      exafel_params.peaksCols = static_cast<uint16_t*>(exafel_peaks_cols.data());
    }
    if(get(options, "sz_threadsafe:exafel:num_peaks", &temp) == pressio_options_key_set) {
      exafel_params.numPeaks=static_cast<uint64_t>(temp);
    }
    if(get(options, "sz_threadsafe:exafel:calib_panel", &exafel_calibPanel) == pressio_options_key_set) {
      exafel_params.calibPanel = static_cast<uint8_t*>(exafel_calibPanel.data());
    }
    get(options, "sz_threadsafe:exafel:tolerance", &exafel_params.tolerance);
    if(get(options, "sz_threadsafe:exafel:bin_size", &temp) == pressio_options_key_set) {
      exafel_params.binSize=static_cast<uint8_t>(temp);
    }
    if(get(options, "sz_threadsafe:exafel:sz_dim", &temp) == pressio_options_key_set) {
      exafel_params.szDim=static_cast<uint8_t>(temp);
    }
    if(get(options, "sz_threadsafe:exafel:peak_size", &temp) == pressio_options_key_set) {
      exafel_params.peakSize=static_cast<uint8_t>(temp);
    }
    if (app == "ExaFEL") {
      user_params = &exafel_params;
    }
}
#endif
#ifdef HAVE_RANDOMACCESS
    get(options, "sz_threadsafe:random_access", &threadsafe_params.randomAccess);
#endif

    return 0;
  }

  int compress_impl(const pressio_data *input, struct pressio_data* output) override {
    compat::shared_lock<compat::shared_mutex> lock(init_handle->sz_init_lock);
    size_t r1 = pressio_data_get_dimension(input, 0);
    size_t r2 = pressio_data_get_dimension(input, 1);
    size_t r3 = pressio_data_get_dimension(input, 2);
    size_t r4 = pressio_data_get_dimension(input, 3);
    size_t r5 = pressio_data_get_dimension(input, 4);
    int status = SZ_NSCS;
    size_t outsize = 0;

    unsigned char* compressed_data=SZ_compress_customize_threadsafe(app.c_str(),user_params,libpressio_type_to_sz_type(input->dtype()),input->data(),
		    r5,r4,r3,r2,r1,&outsize,&status);

    if(compressed_data != nullptr) {
      *output = pressio_data::move(pressio_byte_dtype, compressed_data, 1, &outsize, pressio_data_libc_free_fn, nullptr);
      return 0;
    } else {
      return set_error(1, "compression failed");
    }
  }
  int decompress_impl(const pressio_data *input, struct pressio_data* output) override {
    compat::shared_lock<compat::shared_mutex> lock(init_handle->sz_init_lock);

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
    void* decompressed_data=SZ_decompress_customize_threadsafe(
        app.c_str(),
        user_params,
        libpressio_type_to_sz_type(output->dtype()),
        static_cast<unsigned char*>(input->data()),
        pressio_data_get_dimension(input, 0),
		    r[4],
        r[3],
        r[2],
        r[1],
        r[0],
        &status);
    
    if(decompressed_data != nullptr) {
      *output = pressio_data::move(type, decompressed_data, ndims, r, pressio_data_libc_free_fn, nullptr);
      return 0;
    } else {
      return set_error(2, "decompression failed");
    }
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
    return "sz_threadsafe";
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override {
	return compat::make_unique<sz_threadsafe_plugin>(*this);
  }
  static const std::string sz_version;
  std::string app = "SZ";
  void* user_params = &threadsafe_params;
  std::shared_ptr<sz_init_handle> init_handle;
#if PRESSIO_SZ_VERSION_GREATEREQ(2,1,11,1)
  pressio_data exafel_peaks_segs;
  pressio_data exafel_peaks_rows;
  pressio_data exafel_peaks_cols;
  pressio_data exafel_calibPanel;
  exafelSZ_params exafel_params{};
#endif
};


std::string const sz_threadsafe_plugin::sz_version= get_version_sz_threadsafe(); 

static pressio_register compressor_sz_threadsafe_plugin(compressor_plugins(), "sz_threadsafe", [](){
    return compat::make_unique<sz_threadsafe_plugin>(pressio_get_sz_init_handle()); 
});

} } 
