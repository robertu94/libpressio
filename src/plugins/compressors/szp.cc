#include <map>
#include <string>
#include <algorithm>
#include <iterator>
#include <omp.h>
#include <szp.h>
#include <szp_defines.h>
#include "std_compat/memory.h"
#include "std_compat/optional.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "cleanup.h"

namespace libpressio { namespace compressors { namespace szp_ns {

    const std::map<std::string, int32_t> SZP_ERROR_MODES {
        { "abs", ABS },
        { "rel", REL },
    };
    const std::map<std::string, int32_t> SZP_FAST_MODES {
        { "randomaccess", SZP_RANDOMACCESS },
        { "norandomaccess", SZP_NONRANDOMACCESS },
    };
    std::vector<std::string> options(std::map<std::string, int32_t> const& op) {
        std::vector<std::string> v;
        v.reserve(op.size());
        std::transform(op.begin(), op.end(), std::back_inserter(v), [](
                        typename std::map<std::string, int32_t>::const_reference i
                    ) {
                    return i.first;
                });
        return v;
    }
    std::string option_to_string(std::map<std::string, int32_t> const& op, int32_t value) {
        auto it = std::find_if(op.begin(), op.end(), [value](
                    typename std::map<std::string, int32_t>::const_reference v
                    ){
                    return v.second == value;
                });
        return it->first; 
    }

    int32_t to_dtype(pressio_dtype d) {
        switch(d) {
            case pressio_float_dtype:
                return SZ_FLOAT;
            case pressio_double_dtype:
                return SZ_DOUBLE;
            default:
                throw std::runtime_error("unsupported type in SZp");
        }
    }


class szp_compressor_plugin : public libpressio::compressors::libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:nthreads", n_threads);
    set(options, "pressio:abs", (errBoundMode == ABS) ? absBound : compat::optional<double>());
    set(options, "pressio:rel", (errBoundMode == REL) ? relBound : compat::optional<double>());
    set(options, "szp:abs_bound", absBound);
    set(options, "szp:rel_bound", relBound);
    set(options, "szp:error_mode", errBoundMode);
    set(options, "szp:fast_mode", fastMode);
    set(options, "szp:block_size", block_size);
    set(options, "szp:error_mode_str", option_to_string(SZP_ERROR_MODES, errBoundMode));
    set(options, "szp:fast_mode_str", option_to_string(SZP_FAST_MODES, fastMode));
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    std::vector<std::string> invalidations {}; 
    std::vector<pressio_configurable const*> invalidation_children {}; 
    set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
    set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
    set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));
    set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{}));
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(SZp is an extremely-fast error-bounded lossy compressor.)");
    set(options, "szp:abs_bound", "the absolute error bound");
    set(options, "szp:rel_bound", "the relative error bound");
    set(options, "szp:error_mode", "the error bound mode");
    set(options, "szp:fast_mode", "can be random access or no random access");
    set(options, "szp:error_mode_str", "string for the error bound mode");
    set(options, "szp:fast_mode_str", "string for the fast mode");
    set(options, "szp:block_size", "size for parallel blocks");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "pressio:nthreads", &n_threads);
    get(options, "pressio:abs", &absBound);
    get(options, "pressio:rel", &relBound);
    get(options, "szp:abs_bound", &absBound);
    get(options, "szp:rel_bound", &relBound);
    get(options, "szp:error_mode", &errBoundMode);
    get(options, "szp:fast_mode", &fastMode);
    get(options, "szp:block_size", &block_size);
    std::string tmp_str;
    if(get(options, "szp:error_mode_str", &tmp_str) == pressio_options_key_set) {
        errBoundMode = SZP_ERROR_MODES.at(tmp_str);
    }
    if(get(options, "szp:fast_mode_str", &tmp_str) == pressio_options_key_set) {
        errBoundMode = SZP_FAST_MODES.at(tmp_str);
    }
    return 0;
  }

  int compress_impl(const pressio_data* real_input,
                    struct pressio_data* output) override
  {
    cleanup num_threads;
    if(n_threads > 0) {
        uint32_t old_threads = omp_get_num_threads();
        omp_set_num_threads(n_threads);
        num_threads = [old_threads]{omp_set_num_threads(old_threads);};
    }
    auto input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
    size_t outSize = 0;
    unsigned char *bytes = szp_compress(fastMode, to_dtype(input.dtype()), input.data(), &outSize, errBoundMode, absBound, relBound, input.num_elements(), block_size);  
    *output = pressio_data::move(pressio_byte_dtype, bytes, {outSize}, domain_plugins().build("malloc"));
    return 0;
  }

  int decompress_impl(const pressio_data* real_input,
                      struct pressio_data* output) override
  {
    cleanup num_threads;
    if(n_threads > 0) {
        uint32_t old_threads = omp_get_num_threads();
        omp_set_num_threads(n_threads);
        num_threads = [old_threads]{omp_set_num_threads(old_threads);};
    }
    auto input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
    void* data = (void*)szp_decompress(fastMode, to_dtype(output->dtype()), reinterpret_cast<unsigned char*>(input.data()), input.num_elements(), output->num_elements(), block_size);
    *output = pressio_data::move(output->dtype(), data, output->dimensions(), domain_plugins().build("malloc"));
    return 0;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "szp"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<szp_compressor_plugin>(*this);
  }
  int32_t fastMode = SZP_RANDOMACCESS;
  int32_t errBoundMode = ABS;
  double absBound = 1e-6;
  double relBound = 1e-6;
  int32_t block_size = 64;
  uint32_t n_threads = 1;
};

pressio_register registration(compressor_plugins(), "szp", []() {
  return compat::make_unique<szp_compressor_plugin>();
});

} }
}
