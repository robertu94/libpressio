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

class bg_plugin: public libpressio_compressor_plugin {
  public:
    struct pressio_options get_options_impl() const override {
      struct pressio_options options;
      set(options, "bg:bgMode", bgMode_libpressio);
      set(options, "bg:errorControlMode", errorControlMode_libpressio);
      set(options, "bg:nsd", nsd_libpressio);
      set(options, "bg:dsd", dsd_libpressio);
      return options;
    }

    struct pressio_options get_configuration_impl() const override {
      struct pressio_options options;
      options.set("pressio:thread_safe", static_cast<int>(pressio_thread_safety_multiple));
      return options;
    }

    int set_options_impl(struct pressio_options const& options) override {
      get(options, "bg:bgMode", &bgMode_libpressio);
      get(options, "bg:errorControlMode", &errorControlMode_libpressio);
      get(options, "bg:nsd", &nsd_libpressio);
      get(options, "bg:dsd", &dsd_libpressio);
      return 0;
    }

    int compress_impl(const pressio_data *input, struct pressio_data* output) override {
      int type = libpressio_type_to_bg_type(pressio_data_dtype(input));
      if(type == -1) {
         return set_error(2, "Invalid data type");
      }

      size_t nbEle = static_cast<size_t>(pressio_data_num_elements(input));
      unsigned long outSize;
      void* data = pressio_data_ptr(input, nullptr);
      unsigned char* compressed_data;


      compressed_data = BG_compress_args(type, data, &outSize, bgMode_libpressio, errorControlMode_libpressio, nsd_libpressio, dsd_libpressio, nbEle);

      //that means the compressor is complaining about the parameter
      if(compressed_data == NULL)
      {
        return set_error(2, "Error when bg is compressing the data");
      }

      *output = pressio_data::move(pressio_byte_dtype, compressed_data, 1, &outSize, pressio_data_libc_free_fn, nullptr);
      return 0;
    }

    int decompress_impl(const pressio_data *input, struct pressio_data* output) override {
      int type = libpressio_type_to_bg_type(pressio_data_dtype(output));
      if(type == -1) {
         set_error(2, "Invalid data type");
      }
      unsigned char* bytes = (unsigned char*)pressio_data_ptr(input, nullptr);
      size_t nbEle = static_cast<size_t>(pressio_data_num_elements(output));
      size_t byteLength = pressio_data_get_bytes(input);

      void* decompressed_data = BG_decompress(type, bytes, byteLength, nbEle);
      *output = pressio_data::move(pressio_data_dtype(output), decompressed_data, 1, &nbEle, pressio_data_libc_free_fn, nullptr);
      return 0;
      }


    
    int major_version() const override {
      return 0;
    }
    int minor_version() const override {
      return 0;
    }
    int patch_version() const override {
      return 0;
    }
    int revision_version () const { 
      return 1;
    }

    const char* version() const override {
      return "0.0.1"; 
    }


    const char* prefix() const override {
      return "bg";
    }

    std::shared_ptr<libpressio_compressor_plugin> clone() override {
      return compressor_plugins().build("bg");
    }
  private:
    int internal_error(int rc) { std::stringstream ss; ss << "interal error " << rc; return set_error(1, ss.str()); }
    int reshape_error() { return set_error(2, "failed to reshape array after compression"); }

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
        return -1;
      }
    }

    
    
};

static pressio_register compressor_bg_plugin(compressor_plugins(), "bg", [](){ static auto bg = std::make_shared<bg_plugin>(); return bg; });

