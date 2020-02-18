#include <memory>
#include <string>
#include <sstream>
#include <fpzip.h>
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/printers.h"
#include "libpressio_ext/cpp/pressio.h"

namespace {
  constexpr int INVALID_TYPE = 8;
}

class fpzip_plugin: public libpressio_compressor_plugin {

  struct pressio_options 	get_options_impl () const override {
    struct pressio_options options = pressio_options();
    options.set_type("fpzip:has_header", pressio_option_uint32_type);
    options.set_type("fpzip:prec", pressio_option_uint32_type);
    return options;
  };

  struct pressio_options 	get_configuration_impl () const override {
    struct pressio_options options = pressio_options();
    pressio_options_set_integer(&options, "pressio:thread_safe", pressio_thread_safety_multiple);
    return options;
  };

  int 	set_options_impl (struct pressio_options const& options) override {
    int tmp;
    if( options.get("fpzip:has_header", &tmp) != pressio_options_key_set) {
      has_header = tmp != 0;
    } else {
      return set_error(7, "fpzip:has_header is required");
    }
    if(options.get("fpzip:prec", &prec) != pressio_options_key_set) {
      return set_error(7, "fpzip:prec is required");
    }

    return 0;
  }

  int 	compress_impl (const pressio_data *input, struct pressio_data *output) override {
    int type = pressio_type_to_fpzip_type(input);
    if(type == INVALID_TYPE) {
      return INVALID_TYPE;
    }

    if(!pressio_data_has_data(output))
    {
      *output = pressio_data::owning(pressio_byte_dtype,
          {input->size_in_bytes() + 1024}
          );
    }

    FPZ* fpz = fpzip_write_to_buffer(
        pressio_data_ptr(output, nullptr),
        pressio_data_get_bytes(output)
    );

    fpz->nx = input->get_dimension(0);
    fpz->ny = input->get_dimension(1);
    fpz->nz = input->get_dimension(2);
    fpz->nf = input->get_dimension(3);
    fpz->type = type;
    fpz->prec = prec;

    fpzip_write_to_buffer(output->data(), output->size_in_bytes());
    if(has_header) {
      if(!fpzip_write_header(fpz)) {
        return fpzip_error();
      }
    }
    size_t outsize = fpzip_write(fpz, input->data());
    output->set_dimensions({outsize});

    return 0;
  };

   int 	decompress_impl (const pressio_data *input, struct pressio_data *output) override {
     (void)input;
     (void)output;
     return 0;
   }

   int pressio_type_to_fpzip_type(const pressio_data* data) {
     auto dtype = data->dtype();
     if(dtype == pressio_float_dtype) return 0;
     else if (dtype == pressio_double_dtype) return 1;
     else return invalid_type(dtype);
   }

   int invalid_type(pressio_dtype type) {
     std::stringstream ss;
     ss << "invalid_type: " << type;
     return set_error(INVALID_TYPE, ss.str());
   }

   int fpzip_error() {
    return set_error(fpzip_errno, fpzip_errstr[fpzip_errno]);
   }

  public:

  fpzip_plugin() {
    std::stringstream ss;
    ss << fpzip_plugin::major_version() << "." << fpzip_plugin::minor_version() << "." << fpzip_plugin::patch_version();
    version_str = ss.str();
  }
  
  int	major_version () const override {
    return FPZIP_VERSION_MAJOR;
  }

  int minor_version () const override {
    return FPZIP_VERSION_MINOR;
  }

  int patch_version () const override {
    return FPZIP_VERSION_PATCH;
  }

  const char* version() const override {
    return version_str.c_str();
  }
  const char* prefix() const noexcept override {
    return "fpzip";
  }
  std::shared_ptr<libpressio_compressor_plugin> clone() override {
    return compat::make_unique<fpzip_plugin>(*this);
  }

  private:
  std::string version_str;
  bool has_header;
  int prec;

};

static pressio_register X(compressor_plugins(), "fpzip", [](){ return std::make_shared<fpzip_plugin>(); });
