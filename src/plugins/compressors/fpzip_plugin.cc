#include <memory>
#include <string>
#include <sstream>
#include <algorithm>
#include <fpzip.h>
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/printers.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"

namespace  libpressio { namespace fpzip { 
namespace {
  constexpr int INVALID_TYPE = 8;
}

class fpzip_plugin: public libpressio_compressor_plugin {

  struct pressio_options 	get_options_impl () const override {
    struct pressio_options options = pressio_options();
    set(options, "fpzip:has_header", has_header);
    set(options, "fpzip:prec", prec);
    if(prec == 0) {
      set(options, "pressio:lossless", 1);
    } else {
      set_type(options, "pressio:lossless", pressio_option_int32_type);
    }
    return options;
  };

  struct pressio_options 	get_documentation_impl () const override {
    pressio_options options;
    set(options, "pressio:description", R"(The FPZip lossless floating point compressor.  See also
      Lindstrom, Peter G, and USDOE National Nuclear Security Administration. FPZIP. Computer software.
      Version 1.2.0. June 10, 2017. https://www.osti.gov//servlets/purl/1579935. 
      doi:https://doi.org/10.11578/dc.20191219.2.)");
    set(options, "fpzip:codec_version", "the FPZip Codec version");
    set(options, "fpzip:library_version", "the FPZip library_version");
    set(options, "fpzip:data_model", "the FPZip data model");
    set(options, "fpzip:has_header", "output a header on compression");
    set(options, "fpzip:prec", "the precision to use");

    return options;
  };

  struct pressio_options 	get_configuration_impl () const override {
    pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "stable");
    set(options, "fpzip:codec_version", fpzip_codec_version);
    set(options, "fpzip:library_version", fpzip_library_version);
    set(options, "fpzip:data_model", fpzip_data_model);
    
        std::vector<std::string> invalidations {"fpzip:prec", "fpzip:has_header", "pressio:lossless"}; 
        std::vector<pressio_configurable const*> invalidation_children {}; 
        
        set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
        set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
        set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));
        set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{"pressio:lossless", "fpzip:prec"}));

    return options;
  };

  int 	set_options_impl (struct pressio_options const& options) override {
    int tmp = has_header;
    if( get(options, "fpzip:has_header", &tmp) == pressio_options_key_set) {
      has_header = tmp != 0;
    }

    get(options, "fpzip:prec", &prec);
    int lossless = -1;
    if(get(options, "pressio:lossless", &lossless) == pressio_options_key_set) {
      prec = 0;
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


    auto norm_dim = input->normalized_dims(4,1);
    fpz->nx = static_cast<int>(norm_dim[0]);
    fpz->ny = static_cast<int>(norm_dim[1]);
    fpz->nz = static_cast<int>(norm_dim[2]);
    fpz->nf = static_cast<int>(norm_dim[3]);
    fpz->type = type;
    fpz->prec = prec;

    if(has_header) {
      if(!fpzip_write_header(fpz)) {
        return fpzip_error();
      }
    }
    size_t outsize = fpzip_write(fpz, input->data());
    if(outsize == 0) {
      return fpzip_error();
    }
    fpzip_write_close(fpz);
    output->set_dimensions({outsize});

    return 0;
  };

   int 	decompress_impl (const pressio_data *input, struct pressio_data *output) override {
    int type = pressio_type_to_fpzip_type(output);
    if(type == INVALID_TYPE) {
      return INVALID_TYPE;
    }

    FPZ* fpz = fpzip_read_from_buffer(
        pressio_data_ptr(input, nullptr)
    );
    if(has_header) {
      fpzip_read_header(fpz);
    } else {
      auto norm_dim = output->normalized_dims(4,1);
      fpz->nx = static_cast<int>(norm_dim[0]);
      fpz->ny = static_cast<int>(norm_dim[1]);
      fpz->nz = static_cast<int>(norm_dim[2]);
      fpz->nf = static_cast<int>(norm_dim[3]);
      fpz->type = type;
      fpz->prec = prec;
    }
    size_t read = fpzip_read(fpz, pressio_data_ptr(output, nullptr));
    fpzip_read_close(fpz);
    if(read == 0) {
      return fpzip_error();
    }

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
    static std::string version_str = [this]{
      std::stringstream ss;
      ss << major_version() << '.' << minor_version() << '.' << patch_version();
      return ss.str();
    }();
    return version_str.c_str();
  }
  const char* prefix() const noexcept override {
    return "fpzip";
  }
  std::shared_ptr<libpressio_compressor_plugin> clone() override {
    return compat::make_unique<fpzip_plugin>(*this);
  }

  private:
  int has_header{0};
  int prec{0};

};

static pressio_register compressor_fpzip_plugin(compressor_plugins(), "fpzip", [](){ return std::make_unique<fpzip_plugin>(); });
} }
