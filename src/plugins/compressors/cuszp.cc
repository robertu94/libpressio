#include <cuSZp.h>
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"

namespace libpressio { namespace compressors { namespace cuszp_ns {

cuszp_type_t to_cuszp_type(pressio_data const& d) {
    switch(d.dtype()) {
        case pressio_float_dtype:
            return CUSZP_TYPE_FLOAT;
        case pressio_double_dtype:
            return CUSZP_TYPE_DOUBLE;
        default:
            throw std::runtime_error("unsupported type for cuszp");
    }
}

static std::map<std::string, cuszp_mode_t> cuSZp_MODES {
    {"plain", CUSZP_MODE_PLAIN},
    {"outlier", CUSZP_MODE_OUTLIER},
};
compat::optional<std::string> maybe_mode_str(cuszp_mode_t mode) {
    auto it = std::find_if(std::begin(cuSZp_MODES), std::end(cuSZp_MODES), [mode](auto const& i){
        return i.second == mode;
    });
    if(it != std::end(cuSZp_MODES)) {
        return {it->first};
    } else {
        return compat::nullopt;
    }
}
class cuszp_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:abs", errBound);
    set(options, "cuszp:mode", static_cast<int32_t>(mode));
    set(options, "cuszp:mode_str", maybe_mode_str(mode));
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    std::vector<std::string> modes;
    std::transform(std::begin(cuSZp_MODES), std::end(cuSZp_MODES), std::back_inserter(modes),
                   [](auto const &i) { return i.first; });
    set(options, "cuszp:mode_str", modes);

    std::vector<std::string> invalidations {"pressio:abs", "cuszp:mode_str", "cuszp:mode"}; 
    std::vector<pressio_configurable const*> invalidation_children {}; 

    set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{"pressio:abs"}));



    set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
    set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
    set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));

    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(cuSZp is an ultra-fast and user-friendly GPU error-bounded lossy compressor for floating-point data array (both single- and double-precision). In short, cuSZp has several key features:

    1. Fusing entire compression/decompression phase into one CUDA kernel function.
    2. Efficient latency control and memory access -- targeting ultra-fast end-to-end throughput.
    3. Two encoding modes (plain or outlier modes) supported, high compression ratio for different data patterns. 
    )");
    set(options, "cuszp:mode_str", R"(the mode of the compressor it can be:

    1. plain for a plain-fixed length encoding mode that does not treat outliers specially
    2. outlier for a mode that preserves outliers

    If your dataset is sparse (consisting lots of 0s) -- plain mode will be a good choice; if your dataset exhibits non-sparse and high smoothness -- outlier mode will be a good choice.
    )");
    set(options, "cuszp:mode", "low level option to specify an undocumented mode directly");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "pressio:abs", &errBound);
    std::string mode_str;
    int32_t tmp_mode = 0;
    if(get(options, "cuszp:mode_str", &mode_str) == pressio_options_key_set) {
        if(auto it = cuSZp_MODES.find(mode_str); it !=cuSZp_MODES.end()) {
            mode = it->second;
        } else {
            set_error(1, "unknown mode_str");
        }
    } else if (get(options, "cuszp:mode", &tmp_mode) == pressio_options_key_set) {
        mode = static_cast<cuszp_mode_t>(tmp_mode);
    }
    return 0;
  }

  int compress_impl(const pressio_data* real_input,
                    struct pressio_data* output) override
  {
      try {
          auto input = domain_manager().make_readable(domain_plugins().build("cudamalloc"), *real_input);
          *output = pressio_data::owning(pressio_byte_dtype, {pressio_dtype_size(input.dtype()) *input.num_elements()}, domain_plugins().build("cudamalloc"));
          size_t comp_size = 0;
          cudaStream_t stream;
          cudaStreamCreate(&stream);
          cuSZp_compress(input.data(), static_cast<unsigned char *>(output->data()), input.num_elements(),
                         &comp_size, static_cast<float>(errBound), to_cuszp_type(input), mode, stream);
          cudaStreamDestroy(stream);
          output->reshape({comp_size});
      } catch(std::runtime_error const& ex)  {
          return set_error(1, ex.what());
      }
      return 0;
  }

  int decompress_impl(const pressio_data* real_input,
                      struct pressio_data* real_output) override
  {
      try {
          auto input = domain_manager().make_readable(domain_plugins().build("cudamalloc"), *real_input);
          auto output = domain_manager().make_writeable(domain_plugins().build("cudamalloc"), std::move(*real_output));
          cudaStream_t stream;
          cudaStreamCreate(&stream);
          cudaMemsetAsync(output.data(), 0, output.size_in_bytes(), stream);
          cuSZp_decompress(output.data(), static_cast<unsigned char*>(input.data()), output.num_elements(),
                         input.size_in_bytes(), static_cast<float>(errBound), to_cuszp_type(output), mode, stream);
          cudaStreamDestroy(stream);
          *real_output = std::move(output);
      } catch (std::runtime_error const& ex) {
          return set_error(1, ex.what());
      }
      return 0;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "2.0.0"; }
  const char* prefix() const override { return "cuszp"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<cuszp_compressor_plugin>(*this);
  }

  double errBound = 1e-4;
  cuszp_mode_t mode = CUSZP_MODE_PLAIN;
};

pressio_register registration(compressor_plugins(), "cuszp", []() {
  return compat::make_unique<cuszp_compressor_plugin>();
});

} }
}
