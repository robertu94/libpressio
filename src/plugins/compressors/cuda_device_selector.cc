#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "cleanup.h"
#include <cuda_runtime.h>

namespace libpressio { namespace cuda_device_selector_ns {

class cuda_device_selector_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "cuda_device_selector:compressor", impl_id, impl);
    set(options, "cuda_device_selector:device", device);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    options.copy_from(impl->get_configuration());
    set(options, "pressio:thread_safe", get_threadsafe(*impl));
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_docs(options, "cuda_device_selector:compressor", "which compressor to use", impl);
    set(options, "pressio:description", R"(set the cuda device for GPU compressors)");
    set(options, "cuda_device_selector:device", R"(which cuda device to use for child)");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get_meta(options, "cuda_device_selector:compressor", compressor_plugins(), impl_id, impl);
    get(options, "cuda_device_selector:device", &device);
    return 0;
  }

  int compress_many_impl(compat::span<const pressio_data* const> const& inputs, compat::span<pressio_data*> & outputs) override
  {
    int olddev;
    cudaGetDevice(&olddev);
    cleanup reset_dev([olddev]{cudaSetDevice(olddev);});
    cudaError_t err;
    if((err = cudaSetDevice(device)) != cudaSuccess) {
      return set_error(1, cudaGetErrorString(err));
    }
    int ret = impl->compress_many(inputs.data(), inputs.data() + inputs.size(), 
                               outputs.data(), outputs.data() + outputs.size());
    if(ret) {
      set_error(ret, impl->error_msg());
    }
    return ret;
  }

  int decompress_many_impl(compat::span<const pressio_data* const> const& inputs, compat::span<pressio_data*> & outputs) override
  {
    int olddev;
    cudaGetDevice(&olddev);
    cleanup reset_dev([olddev]{cudaSetDevice(olddev);});
    cudaError_t err;
    if((err = cudaSetDevice(device)) != cudaSuccess) {
      return set_error(1, cudaGetErrorString(err));
    }
    int ret = impl->decompress_many(inputs.data(), inputs.data() + inputs.size(), 
                               outputs.data(), outputs.data() + outputs.size());
    if(ret) {
      set_error(ret, impl->error_msg());
    }
    return ret;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    int olddev;
    cudaGetDevice(&olddev);
    cleanup reset_dev([olddev]{cudaSetDevice(olddev);});
    cudaError_t err;
    if((err = cudaSetDevice(device)) != cudaSuccess) {
      return set_error(1, cudaGetErrorString(err));
    }
    int ret = impl->compress(input, output);
    if(ret) {
      set_error(ret, impl->error_msg());
    }
    return ret;
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    int olddev;
    cudaGetDevice(&olddev);
    cudaError_t err;
    cleanup reset_dev([olddev]{cudaSetDevice(olddev);});
    if((err = cudaSetDevice(device)) != cudaSuccess) {
      return set_error(1, cudaGetErrorString(err));
    }
    int ret = impl->decompress(input, output);
    if(ret) {
      set_error(ret, impl->error_msg());
    }
    return ret;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "cuda_device_selector"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<cuda_device_selector_compressor_plugin>(*this);
  }

  void set_name_impl(std::string const& new_name) override {
    impl->set_name(new_name + '/' + impl->prefix());
  }

  int device = []{ int dev; if(cudaGetDevice(&dev) == cudaSuccess) return dev; else return 0;}();
  std::string impl_id = "noop";
  pressio_compressor impl = compressor_plugins().build(impl_id);

};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "cuda_device_selector", []() {
  return compat::make_unique<cuda_device_selector_compressor_plugin>();
});

} }

