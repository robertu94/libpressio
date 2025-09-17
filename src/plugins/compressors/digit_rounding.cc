#include <cmath> //for exp and log
#include <sstream>
#include <dr/libdround.h>
#include "libpressio_ext/cpp/data.h" //for access to pressio_data structures
#include "libpressio_ext/cpp/compressor.h" //for the libpressio_compressor_plugin class
#include "libpressio_ext/cpp/options.h" // for access to pressio_options
#include "libpressio_ext/cpp/pressio.h" //for the plugin registries
#include "libpressio_ext/cpp/domain_manager.h" //for the plugin registries
#include "pressio_options.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "std_compat/memory.h"

namespace libpressio  { namespace compressors { namespace digit_rounding {

#define INVALID_TYPE -1

class digit_rounding_plugin: public libpressio_compressor_plugin {
  public:
      digit_rounding_plugin() {
      std::stringstream ss;
      ss << digit_rounding_plugin::major_version() << "." << digit_rounding_plugin::minor_version() << "." << digit_rounding_plugin::patch_version() << "." << digit_rounding_plugin::revision_version();
      dround_version = ss.str();
    };
      struct pressio_options get_options_impl() const override {
      struct pressio_options options;
      set(options, "digit_rounding:prec", prec);
      return options;
    }

    struct pressio_options get_configuration_impl() const override {
      struct pressio_options options;
      set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
      set(options, "pressio:stability", "stable");
      
        //TODO fix the list of options for each command
        std::vector<std::string> invalidations {"digit_rounding:prec"}; 
        std::vector<pressio_configurable const*> invalidation_children {}; 
        
        set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
        set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
        set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));

    return options;
    }

    struct pressio_options get_documentation_impl() const override {
      struct pressio_options options;
      set(options, "pressio:description", R"(Various methods for digit rounding which improve compression)");
      set(options, "digit_rounding:prec", "The prec parameter of digit rounding (a number between 0 and 64)");
      return options;
    }


    int set_options_impl(struct pressio_options const& options) override {
      get(options, "digit_rounding:prec", &prec);

      return 0;
    }

    int compress_impl(const pressio_data *real_input, struct pressio_data* output) override {
      pressio_data input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
      int type = libpressio_type_to_dr_type(pressio_data_dtype(&input));
      if(type == INVALID_TYPE) {
         return INVALID_TYPE;
      }

      size_t nbEle = pressio_data_num_elements(&input);
      unsigned long outSize;
      void* data = pressio_data_ptr(&input, nullptr);
      unsigned char* compressed_data = dround_compress(type, data, nbEle, prec, &outSize);

      *output = pressio_data::move(pressio_byte_dtype, compressed_data, 1, &outSize, pressio_data_libc_free_fn, nullptr);
      return 0;
    }

    int decompress_impl(const pressio_data *real_input, struct pressio_data* output) override {
      pressio_data input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
      int type = libpressio_type_to_dr_type(pressio_data_dtype(output));
      if(type == INVALID_TYPE) {
         return INVALID_TYPE;
      }
      void* bytes = pressio_data_ptr(&input, nullptr);
      size_t nbEle = pressio_data_num_elements(output);
      size_t outSize = input.size_in_bytes();
      void* decompressed_data = dround_decompress(type, (unsigned char*)bytes, nbEle, static_cast<unsigned long>(outSize));
      *output = pressio_data::move(pressio_data_dtype(output), decompressed_data, 1, &nbEle, pressio_data_libc_free_fn, nullptr);
      return 0;
      }


    
    int major_version() const override {
      return DROUND_VER_MAJOR;
    }
    int minor_version() const override {
      return DROUND_VER_MINOR;
    }
    int patch_version() const override {
      return DROUND_VER_BUILD;
    }
    int revision_version () const { 
      return DROUND_VER_REVISION;
    }

    const char* version() const override {
      return dround_version.c_str(); 
    }


    const char* prefix() const override {
      return "digit_rounding";
    }

    std::shared_ptr<libpressio_compressor_plugin> clone() override {
      return compat::make_unique<digit_rounding_plugin>(*this);
    }
  private:
    int prec;
    std::string dround_version;
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
        set_error(2, "Invalid data type");
        return INVALID_TYPE;
      }
    }

    
    
};

pressio_register registration(compressor_plugins(), "digit_rounding", [](){return compat::make_unique<digit_rounding_plugin>(); });
} }}
