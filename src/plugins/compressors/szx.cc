#include <sstream>
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "cleanup.h"
#include <szx.h>
#include <omp.h>
#include <pressio_version.h>
#ifdef LIBPRESSIO_HAS_CUSZx
#include <cuszx_entry.h>
#endif

namespace libpressio { namespace szx_ns {


  const std::map<std::string,int> ERR_MODES {
    {"abs", ABS},
    {"rel", REL},
    {"fxr", FXR},
  };
  const std::map<std::string,int> COMP_MODES {
    {"no_block_fast", SZx_NO_BLOCK_FAST_CMPR},
    {"block_fast", SZx_WITH_BLOCK_FAST_CMPR},
    {"randomaccess_fast", SZx_RANDOMACCESS_FAST_CMPR},
    {"openmp_fast", SZx_OPENMP_FAST_CMPR},
  };

  std::vector<std::string> keys(std::map<std::string, int> const& entries) {
    std::vector<std::string> k;
    k.reserve(entries.size());
    for (auto const& i : entries) {
      k.emplace_back(i.first);
    }
    return k;
  }

class szx_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    if(errBoundMode == ABS){
      set(options, "pressio:abs", absErrBound);
    } else {
      set_type(options, "pressio:abs", pressio_option_double_type);
    }
    if(errBoundMode == REL){
      set(options, "pressio:rel", relBoundRatio);
    } else {
      set_type(options, "pressio:rel", pressio_option_double_type);
    }

    set(options, "pressio:nthreads", nthreads);
    set(options, "szx:nthreads", nthreads);
    set(options, "szx:abs_err_bound", absErrBound);
    set(options, "szx:rel_bound_ratio", relBoundRatio);
    set(options, "szx:tolerance", tolerance);
    set(options, "szx:fixed_ratio", ratio);
    set(options, "szx:fast_mode", fastMode);
    set_type(options, "szx:fast_mode_str", pressio_option_charptr_type);
    set(options, "szx:err_bound_mode", errBoundMode);
    set_type(options, "szx:err_bound_mode_str", pressio_option_charptr_type);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    set(options, "szx:err_bound_mode_str", keys(ERR_MODES));
    set(options, "szx:fast_mode_str", keys(COMP_MODES));


      set(options, "predictors:error_dependent", std::vector<std::string>{
    "pressio:abs",
    "szx:tolerance",
    "szx:fixed_ratio",
    "szx:fast_mode",
    "szx:fast_mode_str",
    "szx:abs_err_bound",
    "szx:rel_bound_ratio",
    "szx:err_bound_mode",
    "szx:err_bound_mode_str",
    });
      set(options, "predictors:error_agnostic", std::vector<std::string>{
    "pressio:abs",
    "szx:tolerance",
    "szx:fixed_ratio",
    "szx:fast_mode",
    "szx:fast_mode_str",
    "szx:abs_err_bound",
    "szx:rel_bound_ratio",
    "szx:err_bound_mode",
    "szx:err_bound_mode_str",
    });

    set(options, "predictors:runtime", std::vector<std::string>{ "szx:nthreads" });


    
        set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", {}, std::vector<std::string>{"pressio:abs", "pressio:rel", "pressio:nthreads"}));

    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(an ultra fast error bounded lossy compressor)");
    set(options, "szx:nthreads", "number of threads to use for szx");
    set(options, "szx:abs_err_bound", "absolute pointwise error bound");
    set(options, "szx:rel_bound_ratio", "pointwise relative error bound error bound");
    set(options, "szx:tolerance", "tolerance of the fixed compression ratio");
    set(options, "szx:fixed_ratio", "target compression ratio for fixed ratio (fxr) mode");
    set(options, "szx:fast_mode", "compression approach");
    set(options, "szx:fast_mode_str", "compression approach as a human readable string");
    set(options, "szx:err_bound_mode", "error bound type");
    set(options, "szx:err_bound_mode_str", "error bound type as a human readable string");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    try {
      if(get(options, "pressio:abs", &absErrBound) == pressio_options_key_set) {
        errBoundMode = ABS;
      }
      if(get(options, "pressio:rel", &relBoundRatio) == pressio_options_key_set) {
        errBoundMode = REL;
      }
      if(get(options, "pressio:nthreads", &nthreads) == pressio_options_key_set) {
        fastMode= COMP_MODES.at("openmp_fast");
      }
      get(options, "szx:tolerance", &tolerance);
      get(options, "szx:fixed_ratio", &ratio);
      get(options, "szx:nthreads", &nthreads);
      
      get(options, "szx:fast_mode", &fastMode);
      std::string tmp;
      if(get(options, "szx:fast_mode_str", &tmp) == pressio_options_key_set) {
        fastMode = COMP_MODES.at(tmp);
      }
      get(options, "szx:abs_err_bound", &absErrBound);
      get(options, "szx:rel_bound_ratio", &relBoundRatio);
      get(options, "szx:err_bound_mode", &errBoundMode);
      if(get(options, "szx:err_bound_mode_str", &tmp) == pressio_options_key_set) {
        errBoundMode = ERR_MODES.at(tmp);
      }
    } catch (std::out_of_range const& ex) {
      return set_error(1, ex.what());
    }
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    try {
    cleanup cleanup_nthreads;
    if(nthreads) {
        auto old_threads = omp_get_num_threads();
        omp_set_num_threads(static_cast<int>(nthreads));
        cleanup_nthreads = [old_threads]() {
            omp_set_num_threads(old_threads);
        };
    }
    size_t outsize = 0;
    auto dims = input->normalized_dims(5);
    unsigned char* bytes = nullptr;
#if LIBPRESSIO_HAS_CUSZx
    if (use_cuda) {
        if(input->dtype() != pressio_float_dtype) {
            return set_error(1, "unsupported type");
        }
        bytes = cuSZx_fast_compress_args_unpredictable_blocked_float(
                static_cast<float*>(input->data()),
                &outsize,
                static_cast<float>(absErrBound),
                input->num_elements(),
                blockSize);
    } else {
#endif
    bytes = SZx_fast_compress_args(
        fastMode,
        to_szx_type(input->dtype()),
        input->data(),
        &outsize,
        errBoundMode,
        static_cast<float>(absErrBound),
        static_cast<float>(relBoundRatio),
        static_cast<float>(ratio),
        static_cast<float>(tolerance),
        dims[4],
        dims[3],
        dims[2],
        dims[1],
        dims[0]
        );
#if LIBPRESSIO_HAS_CUSZx
    }
#endif
    if(bytes) {
      *output = pressio_data::move(pressio_byte_dtype, bytes, {outsize}, pressio_data_libc_free_fn, nullptr);
      return 0;
    } else {
      return set_error(2, "compression failed");
    }
    } catch(std::runtime_error const& ex) {
      return set_error(1, ex.what());
    }
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    try {
    cleanup cleanup_nthreads;
    if(nthreads) {
        auto old_threads = omp_get_num_threads();
        omp_set_num_threads(static_cast<int>(nthreads));
        cleanup_nthreads = [old_threads]() {
            omp_set_num_threads(old_threads);
        };
    }
    auto dims = output->normalized_dims(5);
    void* output_bytes = nullptr;
#if LIBPRESSIO_HAS_CUSZx
    if(use_cuda) {

        cuSZx_fast_decompress_args_unpredictable_blocked_float(reinterpret_cast<float**>(&output_bytes), output->num_elements(), static_cast<unsigned char*>(input->data()));
    } else {
#endif
    output_bytes = SZx_fast_decompress(
        fastMode,
        to_szx_type(output->dtype()),
        static_cast<unsigned char*>(input->data()),
        input->size_in_bytes(),
        dims[4],
        dims[3],
        dims[2],
        dims[1],
        dims[0]
        );
#if LIBPRESSIO_HAS_CUSZx
    }
#endif
    if(output) {
      *output = pressio_data::move(output->dtype(), output_bytes, output->dimensions(), pressio_data_libc_free_fn, nullptr);
      return 0;
    } else {
      return set_error(2, "compression failed");
    }
    } catch(std::runtime_error const& ex) {
      return set_error(1, ex.what());
    }
  }

  int major_version() const override { return SZx_VER_MAJOR; }
  int minor_version() const override { return SZx_VER_MINOR; }
  int patch_version() const override { return SZx_VER_BUILD; }
  int revision_version() const { return SZx_VER_REVISION; }
  const char* version() const override { 
    static std::string version_str = [this]{
      std::stringstream ss;
      ss << major_version() << '.' << minor_version() << '.' << patch_version() << '.' << revision_version();
      return ss.str();
    }();
    return version_str.c_str();
  }
  const char* prefix() const override { return "szx"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<szx_compressor_plugin>(*this);
  }

private:
  int to_szx_type(pressio_dtype dtype) {
    switch(dtype) {
      case pressio_float_dtype:
        return SZx_FLOAT;
      case pressio_double_dtype:
        return SZx_DOUBLE;
      default:
        throw std::runtime_error("unsupported type ");

    }
  }

  int32_t fastMode = SZx_WITH_BLOCK_FAST_CMPR;
  int32_t errBoundMode = ABS;
  uint32_t nthreads = 0;
  double absErrBound = 1e-4;
  double relBoundRatio = 0;
  double ratio = 0;
  double tolerance = 0;
#if LIBPRESSIO_HAS_CUSZx
  bool use_cuda = false;
  int32_t blockSize = 0;
#endif
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "szx", []() {
  return compat::make_unique<szx_compressor_plugin>();
});

} }
