#include <cmath> //for exp and log
#include <sstream>
#include <bg/bg.h>
#include "libpressio_ext/cpp/data.h" //for access to pressio_data structures
#include "libpressio_ext/cpp/compressor.h" //for the libpressio_compressor_plugin class
#include "libpressio_ext/cpp/options.h" // for access to pressio_options
#include "libpressio_ext/cpp/pressio.h" //for the plugin registries
#include "pressio_options.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "libpressio_ext/compat/memory.h"

#define VERSION_PATCH 3
#define INVALID_TYPE -1


class bit_grooming_plugin: public libpressio_compressor_plugin {
  public:
    struct pressio_options get_options_impl() const override {
      struct pressio_options options;
      set(options, "bit_grooming:bgMode", bgMode_libpressio);
      set(options, "bit_grooming:errorControlMode", errorControlMode_libpressio);
      set(options, "bit_grooming:nsd", nsd_libpressio);  //number of significant digits
      set(options, "bit_grooming:dsd", dsd_libpressio);  //number of significant decimal digits
      return options;
    }

    struct pressio_options get_configuration_impl() const override {
      struct pressio_options options;
      options.set("pressio:thread_safe", static_cast<int>(pressio_thread_safety_multiple));
      return options;
    }

    int set_options_impl(struct pressio_options const& options) override {
      get(options, "bit_grooming:bgMode", &bgMode_libpressio);
      get(options, "bit_grooming:errorControlMode", &errorControlMode_libpressio);
      get(options, "bit_grooming:nsd", &nsd_libpressio);  //number of significant digits
      get(options, "bit_grooming:dsd", &dsd_libpressio);  //number of significant decimal digits
      return 0;
    }

    int compress_impl(const pressio_data *input, struct pressio_data* output) override {
      int type = libpressio_type_to_bg_type(pressio_data_dtype(input));
      if(type == INVALID_TYPE) {
         return INVALID_TYPE;
      }

      size_t nbEle = pressio_data_num_elements(input);
      unsigned long outSize;
      void* data = pressio_data_ptr(input, nullptr);
      unsigned char* compressed_data;


      compressed_data = BG_compress_args(type, data, &outSize, bgMode_libpressio, errorControlMode_libpressio, nsd_libpressio, dsd_libpressio, nbEle);

      if(compressed_data == NULL)
      {
        return set_error(2, "Error when bit grooming is compressing the data");
      }

      *output = pressio_data::move(pressio_byte_dtype, compressed_data, 1, &outSize, pressio_data_libc_free_fn, nullptr);
      return 0;
    }

    int decompress_impl(const pressio_data *input, struct pressio_data* output) override {
      int type = libpressio_type_to_bg_type(pressio_data_dtype(output));
      if(type == INVALID_TYPE) {
         return INVALID_TYPE;
      }
      unsigned char* bytes = (unsigned char*)pressio_data_ptr(input, nullptr);
      size_t nbEle = pressio_data_num_elements(output);
      size_t byteLength = pressio_data_get_bytes(input);

      void* decompressed_data = BG_decompress(type, bytes, byteLength, nbEle);
      *output = pressio_data::move(pressio_data_dtype(output), decompressed_data, 1, &nbEle, pressio_data_libc_free_fn, nullptr);
      return 0;
      }


    
    int major_version() const override {
      return BG_VER_MAJOR;
    }
    int minor_version() const override {
      return BG_VER_MINOR;
    }
    int patch_version() const override {
      return VERSION_PATCH;
    }
    int revision_version () const { 
      return BG_VER_REVISION;
    }

    const char* version() const override {
      return "0.0.1"; 
    }


    const char* prefix() const override {
      return "bit_grooming";
    }

    std::shared_ptr<libpressio_compressor_plugin> clone() override {
      return compat::make_unique<bit_grooming_plugin>(*this);
    }
  private:
    int libpressio_type_to_bg_type(pressio_dtype type)
    {
      if(type == pressio_float_dtype)
      {
        return BG_FLOAT;
      }
      else if(type == pressio_double_dtype)
      {
        return BG_FLOAT;
      }
      else
      {
        set_error(2, "Invalid data type")
        return INVALID_TYPE;
      }
    }

    
    
};

static pressio_register compressor_bit_grooming_plugin(compressor_plugins(), "Bit Grooming", [](){ static auto bg = std::make_shared<bit_grooming_plugin>(); return bg; });

