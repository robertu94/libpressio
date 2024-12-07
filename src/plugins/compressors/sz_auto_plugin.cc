#include <sstream>
#include "sz_autotuning_3d.hpp"
#include "sz_def.hpp"
#include "sz_utils.hpp"
#include "libpressio_ext/cpp/data.h" //for access to pressio_data structures
#include "libpressio_ext/cpp/compressor.h" //for the libpressio_compressor_plugin class
#include "libpressio_ext/cpp/options.h" // for access to pressio_options
#include "libpressio_ext/cpp/pressio.h" //for the plugin registries
#include "libpressio_ext/cpp/domain_manager.h" //for the plugin registries
#include "pressio_options.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "std_compat/memory.h"

namespace libpressio { namespace sz_auto {

class sz_auto_plugin: public libpressio_compressor_plugin {
  public:
    sz_auto_plugin() {
      std::stringstream ss;
      ss << sz_auto_plugin::major_version() << "." << sz_auto_plugin::minor_version() << "." << sz_auto_plugin::patch_version() << "." << sz_auto_plugin::revision_version();
      sz_auto_version = ss.str();
    };
    struct pressio_options get_options_impl() const override {
      struct pressio_options options;
      set(options, "pressio:abs", error_bounds);
      set(options, "SZauto:error_bounds", error_bounds);
      set(options, "SZauto:sample_ratio", sample_ratio);
      return options;
    }

    struct pressio_options get_configuration_impl() const override {
      struct pressio_options options;
      set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
      set(options,"pressio:stability", "experimental");
      return options;
    }

    struct pressio_options get_documentation_impl() const override {
      struct pressio_options options;
      set(options, "pressio:description", R"(SZ C++ Version that Supports Second-Order Prediction and Parameter 
          Optimization. 

          See also SZauto: Kai Zhao, Sheng Di, Xin Liang, Sihuan Li, Dingwen Tao, Zizhong Chen,
          and Franck Cappello. "Significantly Improving Lossy Compression for HPC Datasets with 
          Second-Order Prediction and Parameter Optimization", Proceedings of the 29th International Symposium on 
          High-Performance Parallel and Distributed Computing (HPDC 20), Stockholm, Sweden, 2020. )");
      set(options, "SZauto:error_bounds", "the absolute error bound to apply");
      set(options, "SZauto:sample_ratio", "the sampling ratio used for tuning");
      return options;
    }


    int set_options_impl(struct pressio_options const& options) override {
      get(options, "pressio:abs", &error_bounds);
      get(options, "SZauto:error_bounds", &error_bounds);
      get(options, "SZauto:sample_ratio", &sample_ratio);
      return 0;
    }

    int compress_impl(const pressio_data *real_input, struct pressio_data* output) override {
      pressio_data input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
      enum pressio_dtype type = pressio_data_dtype(&input);
      unsigned long outSize;
      void* data = pressio_data_ptr(&input, nullptr);
      unsigned char* compressed_data;

      size_t ndims = pressio_data_num_dimensions(&input);
      size_t r1 = pressio_data_get_dimension(&input, 0);
      size_t r2 = pressio_data_get_dimension(&input, 1);
      size_t r3 = pressio_data_get_dimension(&input, 2);

      if(ndims != 3)
      {
        return set_error(2, "Error: SZauto only supports 3d compression");
      }

      if(type == pressio_float_dtype)
      {
        compressed_data = sz_compress_autotuning_3d<float>((float*)data, r1, r2, r3, error_bounds, outSize, false, false, false, sample_ratio);
      }
      else if(type == pressio_double_dtype)
      {
        compressed_data = sz_compress_autotuning_3d<double>((double*)data, r1, r2, r3, error_bounds, outSize, false, false, false, sample_ratio);
      }
      else
      {
        return set_error(2, "Error: SZauto only supports float or double");
      }

      //that means the compressor is complaining about the parameter
      if(compressed_data == NULL)
      {
        return set_error(2, "Error when SZauto is compressing the data");
      }

      *output = pressio_data::move(pressio_byte_dtype, compressed_data, 1, &outSize, pressio_data_libc_free_fn, nullptr);
      return 0;
    }

    int decompress_impl(const pressio_data *real_input, struct pressio_data* output) override {
      enum pressio_dtype dtype = pressio_data_dtype(output);

      size_t r[] = {
        pressio_data_get_dimension(output, 0),
        pressio_data_get_dimension(output, 1),
        pressio_data_get_dimension(output, 2),
      };
      size_t ndims = pressio_data_num_dimensions(output);
      if(ndims != 3)
      {
        return set_error(2, "Error: SZauto only supports 3d decompression");
      }

      pressio_data input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);

      size_t compressed_size;
      void* compressedBytes = pressio_data_ptr(&input, &compressed_size);

      void* decompressed_data;
      if(dtype == pressio_float_dtype)
      {
        decompressed_data = sz_decompress_autotuning_3d<float>((unsigned char*)compressedBytes, compressed_size, r[0], r[1], r[2]);
      }
      else if(dtype == pressio_double_dtype)
      {
        decompressed_data = sz_decompress_autotuning_3d<double>((unsigned char*)compressedBytes, compressed_size, r[0], r[1], r[2]);
      }
      else
      {
        return set_error(2, "Error: SZauto only supports float or double");
      }


      *output = pressio_data::move(dtype, decompressed_data, ndims, r, pressio_data_libc_free_fn, nullptr);
      return 0;
      }


    //the author of SZauto does not release their version info.
    int major_version() const override {
      return SZAUTO_MAJOR_VERSION; 
    }
    int minor_version() const override {
      return SZAUTO_MINOR_VERSION;
    }
    int patch_version() const override {
      return SZAUTO_PATCH_VERSION;
    }
    int revision_version () const { 
      return SZAUTO_REVISION_VERSION;
    }

    const char* version() const override {
      return sz_auto_version.c_str(); 
    }


    const char* prefix() const override {
      return "SZauto";
    }

    std::shared_ptr<libpressio_compressor_plugin> clone() override {
      return compat::make_unique<sz_auto_plugin>(*this);
    }
  private:

    std::string sz_auto_version;
    double error_bounds = 0.1; //params for compressing the sz-auto
    float sample_ratio = 0.05;
};

static pressio_register compressor_sz_auto_plugin(compressor_plugins(), "SZauto", [](){return compat::make_unique<sz_auto_plugin>(); });

} }
