#include <mutex>
#include <szgpu/szgpu.h>
#include <cuda_runtime.h>
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"

namespace libpressio { namespace compressors { namespace szgpu_ns {

// SZ-GPU reuses a small device workspace across calls; calls that share it are
// serialized, and copies of the plugin start without one
struct szgpu_workspace_holder {
    szgpu_workspace_holder() = default;
    szgpu_workspace_holder(szgpu_workspace_holder const&) {}
    szgpu_workspace_holder& operator=(szgpu_workspace_holder const& rhs) {
        if(this != &rhs) reset();
        return *this;
    }
    ~szgpu_workspace_holder() { reset(); }

    szgpu_status_t acquire(size_t n_elements, szgpu_workspace_t** out) {
        int current_device = 0;
        if(cudaGetDevice(&current_device) != cudaSuccess) return SZGPU_STATUS_CUDA_ERROR;
        if(ws == nullptr || device != current_device || szgpu_workspace_capacity(ws) < n_elements) {
            reset();
            szgpu_status_t status = szgpu_workspace_create(&ws, n_elements);
            if(status != SZGPU_STATUS_OK) return status;
            device = current_device;
        }
        *out = ws;
        return SZGPU_STATUS_OK;
    }
    void reset() {
        if(ws) szgpu_workspace_destroy(ws);
        ws = nullptr;
        device = -1;
    }

    std::mutex mutex;
    szgpu_workspace_t* ws = nullptr;
    int device = -1;
};

class szgpu_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:abs", abs_error_bound);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");

    std::vector<std::string> invalidations {"pressio:abs"};
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
    set(options, "pressio:description", R"(SZ-GPU (FSZ) is an extremely fast GPU error-bounded lossy compressor in the SZ family.
    It stands at the family's Pareto frontier: achieving both extreme throughput and a high compression ratio simultaneously.
    See https://github.com/szcompressor/SZ-GPU

    SZ-GPU compresses float and double data under the absolute error bound pressio:abs.
    Inputs may live in host or device memory. Outputs are returned in device memory,
    reusing a caller-provided device buffer when it is large enough.
    Compressing an input without data reports the maximum compressed size.
    Decompression needs the pressio:abs used for compression and an output with the original type and dimensions.
    For a value-range relative bound, use the pressio meta-compressor with pressio:rel.
    )");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "pressio:abs", &abs_error_bound);
    return 0;
  }

  int compress_impl(const pressio_data* real_input,
                    struct pressio_data* output) override
  {
    if(!is_supported(real_input->dtype())) return unsupported_dtype();
    size_t const n = real_input->num_elements();
    size_t const max_bytes = szgpu_max_compressed_bytes(n);
    if(!real_input->has_data()) {
      *output = pressio_data::empty(pressio_byte_dtype, {max_bytes});
      return 0;
    }
    if(!(abs_error_bound > 0)) return set_error(1, "pressio:abs must be positive");

    try {
      auto input = domain_manager().make_readable(domain_plugins().build("cudamalloc"), *real_input);
      *output = domain_manager().make_writeable(domain_plugins().build("cudamalloc"), std::move(*output));
      if(!output->has_data() || output->capacity_in_bytes() < max_bytes) {
        *output = pressio_data::owning(pressio_byte_dtype, {max_bytes}, domain_plugins().build("cudamalloc"));
      }

      std::lock_guard<std::mutex> guard(workspace.mutex);
      szgpu_workspace_t* ws = nullptr;
      szgpu_compress_result_t result{};
      szgpu_status_t status = workspace.acquire(n, &ws);
      if(status == SZGPU_STATUS_OK) {
        unsigned char* d_cmp = static_cast<unsigned char*>(output->data());
        if(input.dtype() == pressio_float_dtype) {
          status = szgpu_compress(static_cast<const float*>(input.data()), d_cmp, n,
                                static_cast<float>(abs_error_bound), ws, 0, &result);
        } else {
          status = szgpu_compress_f64(static_cast<const double*>(input.data()), d_cmp, n,
                                    abs_error_bound, ws, 0, &result);
        }
      }
      if(status != SZGPU_STATUS_OK) return szgpu_error(status);

      output->set_dtype(pressio_byte_dtype);
      output->reshape({result.cmp_size});
    } catch(std::exception const& ex) {
      return set_error(2, ex.what());
    }
    return 0;
  }

  int decompress_impl(const pressio_data* real_input,
                      struct pressio_data* real_output) override
  {
    if(!is_supported(real_output->dtype())) return unsupported_dtype();
    size_t const n = real_output->num_elements();
    if(n == 0) return set_error(1, "the output must have the dimensions of the original data");
    if(!(abs_error_bound > 0)) return set_error(1, "pressio:abs must be positive");

    try {
      auto input = domain_manager().make_readable(domain_plugins().build("cudamalloc"), *real_input);
      szgpu_compress_result_t const result = szgpu_make_result(n, input.size_in_bytes());
      if(input.size_in_bytes() < result.data_offset) {
        return set_error(1, "the compressed data is too small for the output dimensions");
      }
      auto output = domain_manager().make_writeable(domain_plugins().build("cudamalloc"), std::move(*real_output));

      std::lock_guard<std::mutex> guard(workspace.mutex);
      szgpu_workspace_t* ws = nullptr;
      szgpu_status_t status = workspace.acquire(n, &ws);
      if(status == SZGPU_STATUS_OK) {
        const unsigned char* d_cmp = static_cast<const unsigned char*>(input.data());
        if(output.dtype() == pressio_float_dtype) {
          status = szgpu_decompress(static_cast<float*>(output.data()), d_cmp, n,
                                  static_cast<float>(abs_error_bound), &result, ws, 0);
        } else {
          status = szgpu_decompress_f64(static_cast<double*>(output.data()), d_cmp, n,
                                      abs_error_bound, &result, ws, 0);
        }
      }
      cudaError_t err = cudaSuccess;
      if(status == SZGPU_STATUS_OK) err = cudaStreamSynchronize(0);
      *real_output = std::move(output);
      if(status != SZGPU_STATUS_OK) return szgpu_error(status);
      if(err != cudaSuccess) return set_error(SZGPU_STATUS_CUDA_ERROR, cudaGetErrorString(err));
    } catch(std::exception const& ex) {
      return set_error(2, ex.what());
    }
    return 0;
  }

  int major_version() const override { return SZGPU_VERSION_MAJOR; }
  int minor_version() const override { return SZGPU_VERSION_MINOR; }
  int patch_version() const override { return SZGPU_VERSION_PATCH; }
  const char* version() const override { return SZGPU_VERSION_STRING; }
  const char* prefix() const override { return "szgpu"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<szgpu_compressor_plugin>(*this);
  }

private:
  static bool is_supported(pressio_dtype dtype) {
    return dtype == pressio_float_dtype || dtype == pressio_double_dtype;
  }
  int unsupported_dtype() {
    return set_error(1, "szgpu supports only float and double data");
  }
  int szgpu_error(szgpu_status_t status) {
    return set_error(static_cast<int>(status), std::string("szgpu: ") + szgpu_status_string(status));
  }

  double abs_error_bound = 1e-4;
  szgpu_workspace_holder workspace;
};

pressio_register registration(compressor_plugins(), "szgpu", []() {
  return compat::make_unique<szgpu_compressor_plugin>();
});

} } }
