#include <cuSZp/cuSZp_entry.h>
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"

namespace libpressio { namespace cuszp_ns {

class cuszp_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:abs", errBound);

    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(a high-throuhput specialized version of SZ for the GPU)");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "pressio:abs", &errBound);
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
      if(input->dtype() != pressio_float_dtype) {
          return set_error(1, "unsupported type");
      }
      float* in_ptr = (float*)input->data();
      size_t size;
      size_t pad_nbEle = (input->num_elements() + 262144 - 1) / 262144 * 262144;
      if(isDevicePtr(in_ptr)) {
        unsigned char* cmpBytes;
        if(output->has_data() && isDevicePtr(output->data())) {
            cmpBytes = (unsigned char*)output->data();
            SZp_compress_deviceptr(in_ptr, cmpBytes, input->num_elements(), &size, (float)errBound);
            output->set_dtype(pressio_byte_dtype);
            output->set_dimensions({size});
        } else {
            cudaMalloc((void**)&cmpBytes, sizeof(float)*pad_nbEle);
            SZp_compress_deviceptr(in_ptr, cmpBytes, input->num_elements(), &size, (float)errBound);
            *output = pressio_data::owning(pressio_byte_dtype, {size});
            cudaMemcpy(output->data(), cmpBytes, size, cudaMemcpyDeviceToHost);
        }
      } else {
        unsigned char* cmpBytes;
        if(output->has_data()) {
            cmpBytes = (unsigned char*)output->data();
            SZp_compress_hostptr(in_ptr, cmpBytes, input->num_elements(), &size, (float)errBound);
            output->set_dtype(pressio_byte_dtype);
            output->set_dimensions({size});
        } else {
            cmpBytes = (unsigned char*)malloc(input->num_elements()*sizeof(float));
            SZp_compress_hostptr(in_ptr, cmpBytes, input->num_elements(), &size, (float)errBound);
            *output = pressio_data::move(pressio_byte_dtype, cmpBytes, {size}, pressio_data_libc_free_fn, nullptr);
        }
      }
      return 0;
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
      if(output->dtype() != pressio_float_dtype) {
          return set_error(1, "unsupported type");
      }
      unsigned char* in_ptr = (unsigned char*)input->data();
      if(isDevicePtr(in_ptr)) {
        float* decdata;
        if(output->has_data() && isDevicePtr(output->data())) {
            decdata = (float*)output->data();
            SZp_decompress_deviceptr(decdata, in_ptr, output->num_elements(), input->num_elements(), (float)errBound);
        } else {
            cudaMalloc((void**)&decdata, sizeof(float)*output->num_elements());
            SZp_decompress_deviceptr(decdata, in_ptr, output->num_elements(), input->num_elements(), (float)errBound);
            *output = pressio_data::owning(pressio_float_dtype, output->dimensions());
            cudaMemcpy(output->data(), decdata, output->size_in_bytes(), cudaMemcpyDeviceToHost);
        }
      } else {
        float* decdata;
        if(output->has_data()) {
            decdata = (float*)output->data();
            SZp_decompress_hostptr(decdata, in_ptr, output->num_elements(), input->num_elements(), (float)errBound);
        } else {
            decdata = (float*)malloc(input->num_elements()*sizeof(float));
            SZp_decompress_hostptr(decdata, in_ptr, output->num_elements(), input->num_elements(), (float)errBound);
            *output = pressio_data::move(pressio_float_dtype, decdata, output->dimensions(), pressio_data_libc_free_fn, nullptr);
        }
      }
      return 0;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "cuszp1"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<cuszp_compressor_plugin>(*this);
  }
  bool isDevicePtr(void* ptr) const {
    cudaPointerAttributes attrs;
    cudaPointerGetAttributes(&attrs, ptr);
    return (attrs.type == cudaMemoryTypeDevice);
  }

  double errBound;
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "cuszp1", []() {
  return compat::make_unique<cuszp_compressor_plugin>();
});

} }

