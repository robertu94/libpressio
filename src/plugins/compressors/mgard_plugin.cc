#include <memory>
#include <sstream>
#include <mgard_api.h>
#include <mgard_api_float.h>
#include "pressio_data.h"
#include "pressio_options.h"
#include "pressio_compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/printers.h"
#include "libpressio_ext/cpp/pressio.h"

namespace {
enum class mgard_compression_function
{
  tol,
  tol_s,
  tol_qoi_s,
  // not supported no outsize paremter //qoiv_s,
  // not supported no outsize parameter//qoi_s,
  tol_normqoi_s,
  invalid,
};
enum class mgard_decompression_function
{
  no_s,
  s,
};

template <class Type>
Type
get_converted(pressio_option const& arg)
{
  assert(arg.has_value());
  return arg.as(pressio_type_to_enum<Type>(), pressio_conversion_explicit).template get_value<Type>();
  }
}

class mgard_plugin: public libpressio_compressor_plugin {

  struct pressio_options* get_configuration_impl() const override {
    struct pressio_options* options = pressio_options_new();
    pressio_options_set_integer(options, "pressio:thread_safe", pressio_thread_safety_single);
    return options;
  }

  struct pressio_options * 	get_options_impl () const override {
    struct pressio_options* options = pressio_options_new();

    auto set_if_set = [options](const char* key, pressio_option_type type, pressio_option const& option) {
      if(option.has_value()) {
        options->set(key, option);
      } else {
        options->set_type(key, type);
      }
    };
    set_if_set("mgard:tolerance", pressio_option_double_type, tolerance);
    set_if_set("mgard:s", pressio_option_double_type, s);
    set_if_set("mgard:norm_of_qoi", pressio_option_double_type, qoi_double);
    set_if_set("mgard:qoi_double", pressio_option_userptr_type, qoi_double);
    set_if_set("mgard:qoi_float", pressio_option_userptr_type, qoi_float);
    return options;
  };

  int 	set_options_impl (struct pressio_options const *options) override {
    auto set_fn = [options](const char* key, pressio_option& option_out) {
      if(options->key_status(key) == pressio_options_key_set) {
        option_out = options->get(key);
      }
    };
    set_fn("mgard:tolerance", tolerance);
    set_fn("mgard:s", s);
    set_fn("mgard:norm_of_qoi", qoi_double);
    set_fn("mgard:qoi_double", qoi_double);
    set_fn("mgard:qoi_float", qoi_float);
    return 0;
  }

  int 	compress_impl (const pressio_data *input, struct pressio_data *output) override {
    {
      int rc = check_configuration(input);
      if( rc != 0) {
        return rc;
      } 
    }
    //mgard destroys the input so we must copy it here to prevent the real input from being destroyed
    auto type = pressio_data_dtype(input); 
    auto input_copy = pressio_data::clone(*input);
    switch(type) {
        case pressio_double_dtype:
          *output = compress_typed<double>(std::move(input_copy));
          return 0;
        case pressio_float_dtype:
          *output = compress_typed<float>(std::move(input_copy));
          return 0;
      default:
        return invalid_type(type);
    }
  };

  

   int 	decompress_impl (const pressio_data *input, struct pressio_data *output) override {
     {
      int rc = check_configuration(output);
      if( rc != 0) {
        return rc;
      } 
     }
    auto input_copy = pressio_data::clone(*input);
    auto type = pressio_data_dtype(output); 
    switch(type) {
        case pressio_double_dtype:
          decompress_typed<double>(std::move(input_copy), output);
          return 0;
        case pressio_float_dtype:
          decompress_typed<float>(std::move(input_copy), output);
          return 0;
      default:
        return invalid_type(type);
    }


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

  /**
   * typed dispatch function for compression
   *
   * \param[in] input_data the data to be compressed, it will be destroyed by
   * this function, so make a copy first
   * \returns a pressio_data with the compressed data
   */
  template <class InputType>
  pressio_data compress_typed (pressio_data&& input_data) const {
    auto dtype = input_data.dtype();
    auto itype = dtype_to_itype(input_data.dtype());
    std::vector<int> dims(3);
    for (int i = 0; i < 3; ++i) {
      dims.at(i) = input_data.get_dimension(i);
    }

    int outsize = 0;
    mgard_compression_function function = select_compression_function(dtype);
    unsigned char* compressed_bytes;
    using qoi_fn =  InputType (*)(int, int, int, InputType*);

    switch(function)
    {
      case mgard_compression_function::tol:
        compressed_bytes = mgard_compress(
            itype,
            static_cast<InputType*>(input_data.data()),
            outsize /*output parameter*/,
            dims.at(0),
            dims.at(1),
            dims.at(2),
            get_converted<InputType>(tolerance)
          );
        break;
      case mgard_compression_function::tol_s:
        compressed_bytes = mgard_compress(
            itype,
            static_cast<InputType*>(input_data.data()),
            outsize /*output parameter*/,
            dims.at(0),
            dims.at(1),
            dims.at(2),
            get_converted<InputType>(tolerance),
            get_converted<InputType>(s)
          );
        break;
      case mgard_compression_function::tol_qoi_s:
        compressed_bytes = mgard_compress(
            itype,
            static_cast<InputType*>(input_data.data()),
            outsize /*output parameter*/,
            dims.at(0),
            dims.at(1),
            dims.at(2),
            get_converted<InputType>(tolerance),
            reinterpret_cast<qoi_fn>(s.get_value<void*>()),
            get_converted<InputType>(s)
          );
        break;
      case mgard_compression_function::tol_normqoi_s:
        compressed_bytes = mgard_compress(
            itype,
            static_cast<InputType*>(input_data.data()),
            outsize /*output parameter*/,
            dims.at(0),
            dims.at(1),
            dims.at(2),
            get_converted<InputType>(tolerance),
            get_converted<InputType>(norm_of_qoi),
            get_converted<InputType>(s)
          );
        break;
      case mgard_compression_function::invalid:
        compressed_bytes = nullptr;
        outsize = 0;
    }

    return pressio_data::move(
        dtype,
        compressed_bytes,
        std::vector<size_t>{static_cast<size_t>(outsize)},
        pressio_data_libc_free_fn,
        nullptr
        );
  }
  
  template <class InputType>
  int decompress_typed (pressio_data&& input_data, pressio_data* output_data) const {
    mgard_decompression_function function = select_decompression_function();
    auto itype = dtype_to_itype(output_data->dtype());
    InputType* output_buffer = nullptr;
    InputType quantizer = 0; /*unused by the mgard as far as I can tell, but part of the signature*/
    std::vector<int> dims(output_data->num_dimensions());
    for (int i = 0; i < 3; ++i) {
      dims[i] = output_data->get_dimension(i);
    }

    switch(function)
    {
      case mgard_decompression_function::s:
        output_buffer = mgard_decompress(
            itype,
            quantizer,
            static_cast<unsigned char*>(input_data.data()),
            static_cast<int>(input_data.size_in_bytes()),
            dims.at(0),
            dims.at(1),
            dims.at(2),
            get_converted<InputType>(s)
            );
        break;
      case mgard_decompression_function::no_s:
        output_buffer = mgard_decompress(
            itype,
            quantizer,
            static_cast<unsigned char*>(input_data.data()),
            static_cast<int>(input_data.size_in_bytes()),
            dims.at(0),
            dims.at(1),
            dims.at(2)
            );
        break;
    }

    *output_data = pressio_data::move(
        output_data->dtype(),
        output_buffer,
        output_data->dimensions(),
        pressio_data_libc_free_fn,
        nullptr
        );
    return 0;
  }

  /**
   * select a compression function
   * \param[in] type which type to use
   * \returns which mgard function should be used for compression
   */
  mgard_compression_function select_compression_function(pressio_dtype type) const {
    if(tolerance.has_value() && !s.has_value() && !qoi_typed(type).has_value() && !norm_of_qoi.has_value()) return mgard_compression_function::tol;
    else if(tolerance.has_value() && s.has_value() && !qoi_typed(type).has_value() && !norm_of_qoi.has_value()) return mgard_compression_function::tol_s;
    else if(tolerance.has_value() && s.has_value() && qoi_typed(type).has_value() && !norm_of_qoi.has_value()) return mgard_compression_function::tol_qoi_s;
    else if(tolerance.has_value() && s.has_value() && !qoi_typed(type).has_value() && norm_of_qoi.has_value()) return mgard_compression_function::tol_normqoi_s;
    else return mgard_compression_function::invalid;
  }

  /**
   * select a decompression function
   * \returns which mgard function should be used for compression
   */
  mgard_decompression_function select_decompression_function() const {
    if(s.has_value()) return mgard_decompression_function::s;
    else return mgard_decompression_function::no_s;
  }

  /**
   * convert pressio_dtypes to mgard itypes
   *
   * \returns itype used in MGARD or -1 on error
   */
  int dtype_to_itype(pressio_dtype type) const {
    if(type == pressio_float_dtype) return 0;
    else if (type == pressio_double_dtype) return 1;
    else return -1;
  }

  /**
   * helper function that returns the appropriate qoi function pointer
   *
   * \param[in] type the type to use for compression
   * \returns which qoi_function to use based on type
   */
  pressio_option const& qoi_typed(pressio_dtype type) const {
    assert(type == pressio_double_dtype || type == pressio_float_dtype);
    if(type == pressio_double_dtype) return qoi_double;
    else return qoi_float;
  }

  /**
   * check configuration of the compressor for both compression and decompression
   * is completely valid, farms out to several checker functions, use this in high-level code
   *
   * \param[in] input the type of the data mgard will process
   * \param[in] type the type of the data mgard will process
   * \returns 0 if the configuration is valid
   */
  int
  check_configuration(pressio_data const* input)
  {
    pressio_dtype type = input->dtype();
    if (!supported_type(type)) {
      return invalid_type(type);
    } else if (!supported_options(type)) {
      return missing_configuration();
    } else if (!supported_dims(pressio_data_num_dimensions(input))) {
      return invalid_dims(input);
    } else
      return 0;
  }

  /**
   * checks if the options used to configure the compressor lead to a valid mgard function
   *
   * \param[in] type type that will be used for compression
   */
  bool supported_options(pressio_dtype type) {
    if(select_compression_function(type) != mgard_compression_function::invalid) return true;
    else return false;
  }

  /**
   * MGARD does not support all types supported by libpressio, check if the passed type is supported
   *
   * \returns false if the type is not supported
   */
  bool supported_type(pressio_dtype type) const {
    switch(type)
    {
    case pressio_float_dtype:
    case pressio_double_dtype:
      return true;
    default:
      return false;
    };
  }

  /**
   * MGARD does not support arbitary dimension data, check if we are using an unsupported dimension
   *
   * \returns false if the dimension is not supported
   */
  bool supported_dims(size_t dims) const {
    switch (dims) {
      case 3:
      case 2:
        return true;
      default:
        return false;
    }
  }

  /**
   * specialization of set_error for configuration parameters that may be missing
   *
   * \returns an error code to propagate
   */
  int missing_configuration() {
    std::stringstream ss;
    ss << "invalid combination of parameters";
    if(tolerance.has_value())
      ss << "tolerance, ";
    if(s.has_value())
      ss << "s, ";
    if(qoi_float.has_value())
      ss << "qoi_float, ";
    if(qoi_double.has_value())
      ss << "qoi_double, ";
    return set_error(1, ss.str());
  }

  /**
   * specialization of set_error for unsupported types
   *
   * \returns an error code to propagate
   */
  int invalid_type(pressio_dtype dtype) {
    std::stringstream ss;
    ss << "invalid type " << dtype;
    return set_error(2, ss.str());
  }

  /**
   * specialization of set_error for unsupported dimensions
   *
   * \returns an error code to propagate
   */
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
  pressio_option qoi_double;
  pressio_option qoi_float;
  pressio_option norm_of_qoi;
};
static pressio_register X(compressor_plugins(), "mgard", [](){ return compat::make_unique<mgard_plugin>(); });
