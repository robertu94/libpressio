#include <memory>
#include <sstream>
#include <mgard_capi.h>
#include "pressio_data.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/printers.h"

namespace {

  template <class... Options>
  bool all_has_value(Options const& ... options) {
    return ((options.has_value()) && ... );
  }
}

class mgard_plugin: public libpressio_compressor_plugin {
  struct pressio_options * 	get_options_impl () const override {
    struct pressio_options* options = pressio_options_new();
    auto set_if_set = [options](const char* key, pressio_option_type type, pressio_option const& option) {
      if(option.has_value()) {
        options->set(key, option);
      } else {
        options->set_type(key, type);
      }
    };
    set_if_set("mgrad:tol", pressio_option_double_type, tolerance);
    set_if_set("mgrad:s", pressio_option_double_type, s);
    set_if_set("mgrad:nfib", pressio_option_int32_type, nfib);
    return options;
  };

  int 	set_options_impl (struct pressio_options const *options) override {
    auto set_fn = [options](const char* key, pressio_option& option_out) {
      if(options->key_status(key) == pressio_options_key_set) {
        option_out.cast_set(options->get(key), pressio_conversion_explicit);
      }
    };
    set_fn("mgrad:tol", tolerance);
    set_fn("mgrad:s", s);
    set_fn("mgrad:nfib", nfib);
    return 0;
  }

  int 	compress_impl (const pressio_data *input, struct pressio_data *output) override {
    auto type = pressio_data_dtype(input);
    if(!all_has_value(tolerance, s, nfib)) {
     return missing_configuration(); 
    } else if(type != pressio_double_dtype) {
      return invalid_type(type);
    } else if(pressio_data_num_dimensions(input) != 2) {
      return invalid_dims(input);
    }

    auto dims = input->dimensions();
    auto input_copy = pressio_data::clone(*input);

    int out_size;
    double tol = tolerance.get_value<double>();
    unsigned char* compressed_data = mgard_compress(
        dtype_to_itype(input_copy.dtype()),
        input_copy.data(),
        &out_size,
        dims.at(0),
        dims.at(1),
        nfib.get_value<int>(),
        &tol,
        s.get_value<double>()
        );
    std::vector<size_t> output_dims = {static_cast<size_t>(out_size)};
    *output = pressio_data::move(pressio_byte_dtype, compressed_data, output_dims, pressio_data_libc_free_fn, nullptr);

    return 0;
  };

   int 	decompress_impl (const pressio_data *input, struct pressio_data *output) override {
     auto type = pressio_data_dtype(output);
     if(!all_has_value(s, nfib)) {
       return missing_configuration();
     } else if(type != pressio_double_dtype) {
       return invalid_type(type);
     } else if(pressio_data_num_dimensions(input) != 2) {
       return invalid_dims(input);
     }

     auto input_copy = pressio_data::clone(*input);


     void* decompressed_data = mgard_decompress(
         dtype_to_itype(output->dtype()),
         static_cast<unsigned char*>(input_copy.data()),
         input->get_dimension(0),
         output->get_dimension(0),
         output->get_dimension(1),
         nfib.get_value<int>(),
         s.get_value<double>()
         );

     *output = pressio_data::move(output->dtype(), decompressed_data, output->dimensions(), pressio_data_libc_free_fn, nullptr);
     return 0;
   }

  public:
  int	major_version () const override {
    return 0;
  }

  int minor_version () const override {
    return 0;
  }

  int patch_version () const override {
    return 0;
  }
  int tweak_version () const {
    return 2;
  }

  const char* version() const override {
    return "0.0.0.2";
  }

  private:
  int dtype_to_itype(pressio_dtype type) const {
    if(type == pressio_float_dtype) return 1;
    else if (type == pressio_double_dtype) return 2;
    else return -1;
  }

  int missing_configuration() {
    std::stringstream ss;
    ss << "missing a required configuration elements";
    if(tolerance.has_value())
      ss << "tolerance, ";
    if(s.has_value())
      ss << "s, ";
    if(nfib.has_value())
      ss << "nfib, ";

    return set_error(1, ss.str());
  }

  int invalid_type(pressio_dtype dtype) {
    std::stringstream ss;
    ss << "invalid type " << dtype;
    return set_error(2, ss.str());
  }

  int invalid_dims(const pressio_data* data) {
    std::stringstream ss;
    ss << "invalid dimentions (" << pressio_data_get_dimension(data, 0);
    for (size_t i = 1; i < pressio_data_num_dimensions(data); ++i) {
      ss << ", " << pressio_data_get_dimension(data, i);
    }
    ss << ")";
    return set_error(3, ss.str());
  }

  pressio_option tolerance;
  pressio_option s;
  pressio_option nfib;
};

std::unique_ptr<libpressio_compressor_plugin> make_c_mgard() {
  return std::make_unique<mgard_plugin>();
}

