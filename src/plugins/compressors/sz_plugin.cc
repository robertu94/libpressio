#include <algorithm>
#include <iterator>
#include <memory>
#include <sstream>
#include <cstdlib>

#include <sz/sz.h>
#if HAVE_WRITESTATS
#include <sz/sz_stats.h>
#endif

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

namespace libpressio { namespace sz {

std::string get_version_sz(){
        static std::string
                s=[]{
			std::stringstream ss;
                        ss << SZ_VER_MAJOR << "." << SZ_VER_MINOR << "." << SZ_VER_BUILD << "." << SZ_VER_REVISION;
                        return ss.str();
                }();
        return s.c_str();
}

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

class sz_plugin: public libpressio_compressor_plugin {
  public:

    sz_plugin(std::shared_ptr<sz_init_handle> && init_handle): init_handle(init_handle) {}

  struct pressio_options get_configuration_impl() const override {
    struct pressio_options options;
    set(options, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_serialized));
    set(options, "pressio:stability", "stable");
#ifdef HAVE_RANDOMACCESS
    set(options, "sz:random_access_enabled", 1u);
#else
    set(options, "sz:random_access_enabled", 0u);
#endif
#ifdef HAVE_TIMECMPR
    set(options, "sz:timecmpr", 1u);
#else
    set(options, "sz:timecmpr", 0u);
#endif
#ifdef HAVE_PASTRI
    set(options, "sz:pastri", 1u);
#else
    set(options, "sz:pastri", 0u);
#endif
#ifdef HAVE_WRITESTATS
    set(options, "sz:write_stats", 1u);
#else
    set(options, "sz:write_stats", 0u);
#endif


    std::vector<std::string> vs;
    std::transform(
        std::begin(sz_mode_str_to_code),
        std::end(sz_mode_str_to_code),
        std::back_inserter(vs),
        [](typename decltype(sz_mode_str_to_code)::const_reference m){ return m.first; });
    set(options, "sz:error_bound_mode_str", vs);
    return options;
  }

  struct pressio_options get_documentation_impl() const override {
    struct pressio_options options;
    set(options, "pressio:description", R"(SZ is an error bounded lossy compressor that uses prediction 
      based methods to compress data. This version of SZ is not threadsafe. Please refer to sz_threadsafe if a threadsafe version of SZ is desired. More information can be found about SZ on its 
      [project homepage](https://github.com/disheng222/sz).)");
    set(options, "sz:random_access_enabled", "true if SZ was compiled in random access mode");
    set(options, "sz:timecmpr", "true if SZ if SZ is compiled in time based compression mode");
    set(options, "sz:pastri", "true if PASTRI mode was built");
    set(options, "sz:write_stats", "true if SZ is compiled with compression statistics support");


    set(options, "sz:abs_err_bound", "the absolute error bound ");
    set(options, "sz:accelerate_pw_rel_compression", "trade compression ratio for a faster pw_rel compression");
    set(options, "sz:app", "access a application specific mode of SZ");
    set(options, "sz:config_file", "filepath passed to SZ_Init()");
    set(options, "sz:config_struct", "structure passed to SZ_Init_Params()" );
    set(options, "sz:data_type", "an internal option to control compression");
    set(options, "sz:error_bound_mode", "integer code used to determine error bound mode");
    set(options, "sz:error_bound_mode_str", "human readable string to set the error bound mode");
    set(options, "sz:exafel:bin_size", "for ROIBIN-SZ, the size of the binning applied");
    set(options, "sz:exafel:calib_panel", "for ROIBIN-SZ the size of the calibration panel");
    set(options, "sz:exafel:num_peaks", "for ROIBIN-SZ the number of peaks");
    set(options, "sz:exafel:peak_size", "for ROIBIN-SZ the size of the peaks");
    set(options, "sz:exafel:peaks_cols", "for ROIBIN-SZ the list of columns peaks appear in");
    set(options, "sz:exafel:peaks_rows", "for ROIBIN-SZ the list of rows peaks appear in");
    set(options, "sz:exafel:peaks_segs", "for ROIBIN-SZ the segments peaks appear in");
    set(options, "sz:exafel:sz_dim", R"(for ROIBIN-SZ the SZ dimensionality prefered
1:  nEvents * panels * pr->binnedRows * pr->binnedCols
2:  nEvents * panels * pr->binnedRows, pr->binnedCols
3:  nEvents * panels, pr->binnedRows, pr->binnedCols
4:  nEvents , pr->binnedRows * panels, pr->binnedCols
    )");
    set(options, "sz:exafel:tolerance", "for ROIBIN-SZ the tolerance used after binning");
    set(options, "sz:gzip_mode", "Which mode to pass to GZIP when used");
    set(options, "sz:lossless_compressor", "Which lossless compressor to use for stage 4");
    set(options, "sz:max_quant_intervals", "the maximum number of quantization intervals");
    set(options, "sz:pred_threshold", "an internal option used to control compression");
    set(options, "sz:prediction_mode", "an internal option used to control compression");
    set(options, "sz:psnr_err_bound", "the bound on the error in the PSNR");
    set(options, "sz:pw_rel_err_bound", "the bound on the pointwise relative error");
    set(options, "sz:quantization_intervals", "the number of quantization intervals to use, 0 means automatic");
    set(options, "sz:random_access", "internal options to use random access mode when compiled in");
    set(options, "sz:rel_err_bound", "the value range relative error bound mode");
    set(options, "sz:sample_distance", "internal option used to control compression");
    set(options, "sz:segment_size", "internal option used to control compression. number of points in each segement for pw_relBoundRatio");
    set(options, "sz:snapshot_cmpr_step", "the frequency of preforming single snapshot based compression in time based compression");
    set(options, "sz:sol_id", "an internal option use d to control compression");
    set(options, "sz:sz_mode", "SZ Mode either SZ_BEST_COMPRESSION or SZ_BEST_SPEED");
    set(options, "sz:user_params", "arguments passed to the application specific mode of SZ. Use in conjunction with sz:app");
    set(options, "sz:protect_value_range", "should the value range be preserved during compression");
    return options;
  }

  struct pressio_options get_options_impl() const override {
    compat::shared_lock<compat::shared_mutex> lock(init_handle->sz_init_lock);
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
#if PRESSIO_SZ_VERSION_GREATEREQ(2,1,11,1)
    set(options, "sz:exafel:peaks_segs", exafel_peaks_segs);
    set(options, "sz:exafel:peaks_rows", exafel_peaks_rows);
    set(options, "sz:exafel:peaks_cols", exafel_peaks_cols);
    set(options, "sz:exafel:num_peaks", static_cast<unsigned int>(exafel_params.numPeaks));
    set(options, "sz:exafel:calib_panel", exafel_calibPanel);
    set(options, "sz:exafel:tolerance", exafel_params.tolerance);
    set(options, "sz:exafel:bin_size", static_cast<unsigned int>(exafel_params.binSize));
    set(options, "sz:exafel:sz_dim", static_cast<unsigned int>(exafel_params.szDim));
    set(options, "sz:exafel:peak_size", static_cast<unsigned int>(exafel_params.peakSize));
#endif
    set(options, "sz:app", app.c_str());
#ifdef HAVE_RANDOMACCESS
    set(options, "sz:random_access", confparams_cpr->randomAccess);
#endif
    set(options, "sz:user_params", user_params);
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override {
    compat::unique_lock<compat::shared_mutex> lock(init_handle->sz_init_lock);

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
#if PRESSIO_SZ_VERSION_GREATEREQ(2,1,11,1)
{
    unsigned int temp;
    if(get(options, "sz:exafel:peaks_segs", &exafel_peaks_segs) == pressio_options_key_set) {
      exafel_params.peaksSegs = static_cast<uint16_t*>(exafel_peaks_segs.data());
    }
    if(get(options, "sz:exafel:peaks_rows", &exafel_peaks_rows) == pressio_options_key_set) {
      exafel_params.peaksRows = static_cast<uint16_t*>(exafel_peaks_rows.data());
    }
    if(get(options, "sz:exafel:peaks_cols", &exafel_peaks_cols) == pressio_options_key_set) {
      exafel_params.peaksCols = static_cast<uint16_t*>(exafel_peaks_cols.data());
    }
    if(get(options, "sz:exafel:num_peaks", &temp) == pressio_options_key_set) {
      exafel_params.numPeaks=static_cast<uint64_t>(temp);
    }
    if(get(options, "sz:exafel:calib_panel", &exafel_calibPanel) == pressio_options_key_set) {
      exafel_params.calibPanel = static_cast<uint8_t*>(exafel_calibPanel.data());
    }
    get(options, "sz:exafel:tolerance", &exafel_params.tolerance);
    if(get(options, "sz:exafel:bin_size", &temp) == pressio_options_key_set) {
      exafel_params.binSize=static_cast<uint8_t>(temp);
    }
    if(get(options, "sz:exafel:sz_dim", &temp) == pressio_options_key_set) {
      exafel_params.szDim=static_cast<uint8_t>(temp);
    }
    if(get(options, "sz:exafel:peak_size", &temp) == pressio_options_key_set) {
      exafel_params.peakSize=static_cast<uint8_t>(temp);
    }
    if (app == "ExaFEL") {
      user_params = &exafel_params;
    }
}
#endif
#ifdef HAVE_RANDOMACCESS
    get(options, "sz:random_access", &confparams_cpr->randomAccess);
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
    if (compressed_data != nullptr) {
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
    return "sz";
  }

  pressio_options get_metrics_results_impl() const override {
    pressio_options sz_metrics;
#if HAVE_WRITESTATS
    set(sz_metrics, "sz:use_mean", sz_stat.use_mean);
    set(sz_metrics, "sz:block_size", (unsigned int)sz_stat.blockSize);
    set(sz_metrics, "sz:lorenzo_blocks", (unsigned int)sz_stat.lorenzoBlocks);
    set(sz_metrics, "sz:regression_blocks", (unsigned int)sz_stat.regressionBlocks);
    set(sz_metrics, "sz:total_blocks", (unsigned int)sz_stat.totalBlocks);
    set(sz_metrics, "sz:huffman_tree_size", (unsigned int)sz_stat.huffmanTreeSize);
    set(sz_metrics, "sz:huffman_coding_size", (unsigned int)sz_stat.huffmanCodingSize);
    set(sz_metrics, "sz:huffman_node_count", (unsigned int)sz_stat.huffmanNodeCount);
    set(sz_metrics, "sz:unpredict_count", (unsigned int)sz_stat.unpredictCount);

    set(sz_metrics, "sz:lorenzo_percent", sz_stat.lorenzoPercent);
    set(sz_metrics, "sz:regression_percent", sz_stat.lorenzoPercent);
    set(sz_metrics, "sz:huffman_compression_ratio", sz_stat.huffmanCompressionRatio);
#endif
    return sz_metrics;
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override {
    return compat::make_unique<sz_plugin>(*this);
  }

  static const std::string sz_version;
  std::string app = "SZ";
  void* user_params = nullptr;
  std::shared_ptr<sz_init_handle> init_handle;
#if PRESSIO_SZ_VERSION_GREATEREQ(2,1,11,1)
  pressio_data exafel_peaks_segs;
  pressio_data exafel_peaks_rows;
  pressio_data exafel_peaks_cols;
  pressio_data exafel_calibPanel;
  exafelSZ_params exafel_params{};
#endif
};

std::unique_ptr<libpressio_compressor_plugin> make_c_sz() {
  return compat::make_unique<sz_plugin>(pressio_get_sz_init_handle());
}

std::string const sz_plugin::sz_version = get_version_sz();

static pressio_register compressor_sz_plugin(compressor_plugins(), "sz", [](){
    return compat::make_unique<sz_plugin>(pressio_get_sz_init_handle()); 
});

} }
