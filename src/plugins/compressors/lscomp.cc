
#include "cuda_runtime.h"
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include <cuLSZ/cuLSZ_entry.h>
#include <cuLSZ/cuLSZ_timer.h>
#include <cuLSZ/cuLSZ_utility.h>



namespace libpressio { namespace compressors { namespace lscomp_ns {

class lscomp_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "lscomp:pooling_threshold", poolingTH);
    set(options, "lscomp:quantization_bins", quantBinsData);
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
    set(options, "pressio:description", R"(A specialized compressors for lightsources focusing on CSSI beamlines
    described in detail in "lsCOMP: Efficient Light Source Compression" published at SC25)");
    set(options, "lscomp:pooling_threshold", "the pooling threhsold");
    set(options, "lscomp:quantization_bins", "the quantization bins for the 4 top levels, x<=y<=z<=w");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "lscomp:pooling_threshold", &poolingTH);
    pressio_data tmp;
    if(get(options, "lscomp:quantization_bins", &tmp) == pressio_options_key_set) {
        tmp = tmp.cast(pressio_uint32_dtype);
        std::vector<uint32_t> v = tmp.to_vector<uint32_t>();
        if(v.size() != 4) return set_error(1, "size of quantization bins must be 4");
        if(!std::is_sorted(v.begin(), v.end())) return set_error(1, "quantization bins values must be increasing");
        quantBinsData = tmp;
    }
    return 0;
  }

  int compress_impl(const pressio_data* real_input,
                    struct pressio_data* output) override
  {
    if(real_input->dtype() != pressio_uint32_dtype || real_input->dtype() != pressio_uint16_dtype) return set_error(1, "unsupported datatype");
    auto input = domain_manager().make_readable(domain_plugins().build("cudamalloc"), *real_input);
    auto dims_sizet = input.normalized_dims(3,1);
    size_t cmpSize = 0;
    uint3 dims {(uint)dims_sizet[0], (uint)dims_sizet[1], (uint)dims_sizet[2]};
    std::vector<uint32_t> quantBinsV = quantBinsData.to_vector<uint32_t>();
    uint4 quantBins {quantBinsV[0], quantBinsV[1], quantBinsV[2], quantBinsV[3]};
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    pressio_data comp_bytes(pressio_data::owning(input.dtype(), {input.num_elements()}));
    switch(input.dtype()) {
        case pressio_uint32_dtype:
            cuLSZ_compression_uint32_bsize64((uint32_t*)input.data(), (uint8_t*)comp_bytes.data(), &cmpSize, dims, quantBins, poolingTH, stream);
            break;
        case pressio_uint16_dtype:
            cuLSZ_compression_uint16_bsize64((uint16_t*)input.data(), (uint8_t*)comp_bytes.data(), &cmpSize, dims, quantBins, poolingTH, stream);
            break;
        default:
            cudaStreamDestroy(stream);
            return set_error(1, "unsupported datatype");
    }
    *output = std::move(comp_bytes);
    output->reshape({cmpSize});
    output->set_dtype(pressio_byte_dtype);
    cudaStreamDestroy(stream);
    *output = std::move(input);
    return 0;
  }

  int decompress_impl(const pressio_data* real_input,
                      struct pressio_data* real_output) override
  {
    auto input = domain_manager().make_readable(domain_plugins().build("cudamalloc"), std::move(*real_input));
    auto output = domain_manager().make_writeable(domain_plugins().build("cudamalloc"), std::move(*real_output));
    auto dims_sizet = output.normalized_dims(3,1);
    uint3 dims {(uint)dims_sizet[0], (uint)dims_sizet[1], (uint)dims_sizet[2]};
    std::vector<uint32_t> quantBinsV = quantBinsData.to_vector<uint32_t>();
    uint4 quantBins {quantBinsV[0], quantBinsV[1], quantBinsV[2], quantBinsV[3]};
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    switch(output.dtype()) {
        case pressio_uint32_dtype:
            cuLSZ_decompression_uint32_bsize64((uint32_t*)output.data(), (uint8_t*)input.data(), input.size_in_bytes(), dims, quantBins, poolingTH, stream);
            break;
        case pressio_uint16_dtype:
            cuLSZ_decompression_uint16_bsize64((uint16_t*)output.data(), (uint8_t*)input.data(), input.size_in_bytes(), dims, quantBins, poolingTH, stream);
            break;
        default:
            cudaStreamDestroy(stream);
            return set_error(1, "unsupported datatype");
    }
    cudaStreamDestroy(stream);
    return 0;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "lscomp"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<lscomp_compressor_plugin>(*this);
  }

  float poolingTH=0.5f;
  pressio_data quantBinsData{3u, 5u, 10u, 15u};
};

pressio_register registration(compressor_plugins(), "lscomp", []() {
  return compat::make_unique<lscomp_compressor_plugin>();
});

} } }

