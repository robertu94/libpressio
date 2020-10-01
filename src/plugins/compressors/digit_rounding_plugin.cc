#include <cmath> //for exp and log
#include <sstream>
#include <dr/libdround.h>
#include "libpressio_ext/cpp/data.h" //for access to pressio_data structures
#include "libpressio_ext/cpp/compressor.h" //for the libpressio_compressor_plugin class
#include "libpressio_ext/cpp/options.h" // for access to pressio_options
#include "libpressio_ext/cpp/pressio.h" //for the plugin registries
#include "pressio_options.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "libpressio_ext/compat/memory.h"

class digit_rounding_plugin: public libpressio_compressor_plugin {
  public:
      struct pressio_options get_options_impl() const override {
      struct pressio_options options;
      set(options, "digit_rounding:prec", prec_user_defined);
      return options;
    }

    struct pressio_options get_configuration_impl() const override {
      struct pressio_options options;
      options.set("pressio:thread_safe", static_cast<int>(pressio_thread_safety_multiple));
      return options;
    }

    int set_options_impl(struct pressio_options const& options) override {
      get(options, "digit_rounding:prec", &prec_user_defined);

      return 0;
    }

    int compress_impl(const pressio_data *input, struct pressio_data* output) override {
      int type = libpressio_type_to_dr_type(pressio_data_dtype(input));
      if(type == -1) {
         return -1;
      }

      size_t nbEle = static_cast<size_t>(pressio_data_num_elements(input));
      unsigned long outSize;
      void* data = pressio_data_ptr(input, nullptr);
      unsigned char* compressed_data = dround_compress_libpressio(type, data, nbEle, &outSize);

      *output = pressio_data::move(pressio_byte_dtype, compressed_data, 1, &outSize, pressio_data_libc_free_fn, nullptr);
      return 0;
    }

    int decompress_impl(const pressio_data *input, struct pressio_data* output) override {
      int type = libpressio_type_to_dr_type(pressio_data_dtype(output));
      if(type == -1) {
         return -1;
      }
      void* bytes = pressio_data_ptr(input, nullptr);
      size_t nbEle = static_cast<size_t>(pressio_data_num_elements(output));
      size_t outSize = input -> size_in_bytes();
      void* decompressed_data = dround_decompress(type, (unsigned char*)bytes, nbEle, static_cast<unsigned long>(outSize));
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
      return "digit_rounding";
    }

    std::shared_ptr<libpressio_compressor_plugin> clone() override {
      return compressor_plugins().build("digit_rounding");
    }
  private:
    int internal_error(int rc) { std::stringstream ss; ss << "interal error " << rc; return set_error(1, ss.str()); }
    int reshape_error() { return set_error(2, "failed to reshape array after compression"); }

    int libpressio_type_to_dr_type(pressio_dtype type)
    {
      if(type == pressio_float_dtype)
      {
        return DIGIT_FLOAT;
      }
      else if(type == pressio_double_dtype)
      {
        return DIGIT_DOUBLE;
      }
      else
      {
        return -1;
      }
    }

    
    
};

static pressio_register compressor_digit_rounding_plugin(compressor_plugins(), "digit_rounding", [](){ static auto dr = std::make_shared<digit_rounding_plugin>(); return dr; });

