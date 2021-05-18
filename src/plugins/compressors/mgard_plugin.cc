#include <cassert>
#include <exception>
#include <memory>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include "std_compat/optional.h"
#include "pressio_version.h"

//some older version of mgard need a float header in addition to the api header
//check for it in CMake and then conditionally include it here if needed
#if LIBPRESSIO_MGARD_NEED_FLOAT_HEADER
#include <mgard_api_float.h>
#endif
#if LIBPRESSIO_MGARD_HAS_CONFIG_HEADER
#include <MGARDConfig.h>
#ifndef MGARD_VERSION_MINOR
#define MGARD_VERSION_MINOR 0
#endif
#ifndef MGARD_VERSION_PATCH
#define MGARD_VERSION_PATCH 0
#endif
#ifndef MGARD_VERSION_TWEAK
#define MGARD_VERSION_TWEAK 0
#endif
#else
//earliest supported version of mgard is 0.0.0.2, but didn't have the version header
#define MGARD_VERSION_STR   "0.0.0.2"
#define MGARD_VERSION_MAJOR 0
#define MGARD_VERSION_MINOR 0
#define MGARD_VERSION_PATCH 0
#define MGARD_VERSION_TWEAK 2
#endif

#define PRESSIO_MGARD_VERSION_GREATEREQ(major, minor, build, revision) \
   (MGARD_VERSION_MAJOR > major || \
   (MGARD_VERSION_MAJOR == major && MGARD_VERSION_MINOR > minor) ||                                  \
   (MGARD_VERSION_MAJOR == major && MGARD_VERSION_MINOR == minor && MGARD_VERSION_PATCH > build) || \
   (MGARD_VERSION_MAJOR == major && MGARD_VERSION_MINOR == minor && MGARD_VERSION_PATCH == build && MGARD_VERSION_TWEAK >= revision))



#include <mgard_api.h>
#if PRESSIO_MGARD_VERSION_GREATEREQ(0,0,0,3)
#include <TensorNorms.tpp>
#endif
#include "pressio_data.h"
#include "pressio_metrics.h"
#include "pressio_options.h"
#include "pressio_compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/printers.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"

struct pressio_mgard_metrics_data {
  std::string metric;
  pressio_metrics metrics;
};

template <class T>
T pressio_mgard_metric(int rows, int cols, int fibs, T* data, void* metrics) {
  auto metric_data = reinterpret_cast<pressio_mgard_metrics_data*>(metrics);
  std::vector<size_t> dims;
  if(rows) {
    dims.push_back(rows);
    if(cols) {
      dims.push_back(cols);
      if(fibs) {
        dims.push_back(fibs);
      }
    }
  }

  pressio_data decompressed = pressio_data::nonowning(pressio_dtype_from_type<T>(), data, dims);
  pressio_options* results = pressio_metrics_evaluate(
      &metric_data->metrics,
      nullptr,
      nullptr,
      &decompressed
      );
  T result = 0.0;
  results->cast(metric_data->metric, &result, pressio_conversion_explicit);
  pressio_options_free(results);


  return result;
}

float pressio_mgard_float(int rows, int cols, int fibs, float* data, void* pressio_metrics_ptr) {
  return pressio_mgard_metric(rows, cols, fibs, data, pressio_metrics_ptr);
}
double pressio_mgard_double(int rows, int cols, int fibs, double* data, void* pressio_metrics_ptr) {
  return pressio_mgard_metric(rows, cols, fibs, data, pressio_metrics_ptr);
}

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

  struct pressio_options get_configuration_impl() const override {
    struct pressio_options options;
    set(options, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_single));
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override {
    struct pressio_options options;
    set(options, "pressio:description",  R"(MGARD is a error bounded lossy compressor based on using multi-level grids.
      More information can be found on onis [project homepage](https://github.com/CODARcode/MGARD))");
    set(options, "mgard:tolerance",  "the tolerance parameter");
    set(options, "mgard:s", "the shape parameter");
    set(options, "mgard:norm_of_qoi", "the norm of the qoi to preserve");
    set(options, "mgard:qoi_double", "function pointer to a double qoi metric");
    set(options, "mgard:qoi_float", "function pointer to a float qoi metric");
#if PRESSIO_MGARD_VERSION_GREATEREQ(0,0,0,3)
    set(options, "mgard:qoi_double_void", "function pointer to a double qoi metric");
    set(options, "mgard:qoi_float_void", "function pointer to a float qoi metric");
    set(options, "mgard:qoi_use_metric", "true if MGARD QOI mode should use a libpressio metric");
    set(options, "mgard:qoi_metric_name", "the id of the metric used for MGARD QOI mode");
#endif
    return options;
  }

  struct pressio_options	get_options_impl () const override {
    struct pressio_options options;
    set(options, "mgard:tolerance",  tolerance);
    set(options, "mgard:s", s);
    set(options, "mgard:norm_of_qoi", norm_of_qoi);
    set(options, "mgard:qoi_double", qoi_double);
    set(options, "mgard:qoi_float", qoi_float);
#if PRESSIO_MGARD_VERSION_GREATEREQ(0,0,0,3)
    set(options, "mgard:qoi_double_void", qoi_double_v);
    set(options, "mgard:qoi_float_void", qoi_float_v);
    set(options, "mgard:qoi_use_metric", qoi_use_metric);
    set(options, "mgard:qoi_metric_name", qoi_metric_name);
#endif
    return options;
  };

  int 	set_options_impl (struct pressio_options const& options) override {
    get(options, "mgard:tolerance", &tolerance);
    auto new_s = s;
    if(get(options, "mgard:s", &new_s) == pressio_options_key_set) {
#if PRESSIO_MGARD_VERSION_GREATEREQ(0,0,0,3)
      if(s != new_s) {
        recompute_metric = true;
      }
#endif
      s = new_s;
    }
    get(options, "mgard:norm_of_qoi", &norm_of_qoi);
    get(options, "mgard:qoi_double", &qoi_double);
    get(options, "mgard:qoi_float", &qoi_float);
#if PRESSIO_MGARD_VERSION_GREATEREQ(0,0,0,3)
    /*
     * prefer qoi is this order:
     *  - use_metric (highest)
     *  - qoi_{float,double}_void
     *  - qoi_{float,double}
     */
    auto new_qoi_double_v = qoi_double_v;
    if(get(options, "mgard:qoi_double_void", &new_qoi_double_v)== pressio_options_key_set) {
      qoi_double = compat::nullopt;
      if(new_qoi_double_v != qoi_double_v) {
        recompute_metric = true;
        qoi_double_v = new_qoi_double_v;
      }
    }
    auto new_qoi_float_v = qoi_float_v;
    if(get(options, "mgard:qoi_float_void", &new_qoi_float_v) == pressio_options_key_set) {
      qoi_float = compat::nullopt;
      if(new_qoi_float_v != qoi_float_v) {
        recompute_metric = true;
        qoi_float_v = new_qoi_float_v;
      }
    }
    auto new_qoi_metric_name = qoi_metric_name;
    if(get(options, "mgard:qoi_metric_name", &new_qoi_metric_name) == pressio_options_key_set) {
      if(new_qoi_metric_name != qoi_metric_name) {
        recompute_metric = true;
        qoi_metric_name = new_qoi_metric_name;
      }
    }
    auto new_qoi_use_metric = qoi_use_metric;
    get(options, "mgard:qoi_use_metric", &new_qoi_use_metric);
    if (new_qoi_use_metric) {
      qoi_use_metric = new_qoi_use_metric;
      qoi_double_v = reinterpret_cast<void*>(pressio_mgard_double);
      qoi_float_v = reinterpret_cast<void*>(pressio_mgard_float);
      qoi_double = compat::nullopt;
      qoi_float = compat::nullopt;
      if (qoi_use_metric != new_qoi_use_metric) {
        recompute_metric = true;
      }
      return 0;
    }
#endif
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
    try {
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
    } catch (std::exception const& e) {
      return set_error(7, e.what());
    }
  };

  

   int 	decompress_impl (const pressio_data *input, struct pressio_data *output) override {
     {
      int rc = check_configuration(output);
      if( rc != 0) {
        return rc;
      } 
     }
    try {
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
    } catch (std::exception const& e) {
      return set_error(7, e.what());
    }


     return 0;
   }

  public:

   pressio_options get_metrics_results_impl() const override {
     pressio_options opts;
#if PRESSIO_MGARD_VERSION_GREATEREQ(0,0,0,3)
     set(opts, "mgard:norm_of_qoi", norm_of_qoi);
     set(opts, "mgard:norm_time", training_time);
#endif
     return opts;
   };

  int	major_version () const override {
    return MGARD_VERSION_MAJOR;
  }

  int minor_version () const override {
    return MGARD_VERSION_MINOR;
  }

  int patch_version () const override {
    return MGARD_VERSION_PATCH;
  }
  int tweak_version () const {
    return MGARD_VERSION_TWEAK;
  }

  const char* version() const override {
    return MGARD_VERSION_STR;
  }

  const char* prefix() const override {
    return "mgard";
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override {
    return compat::make_unique<mgard_plugin>(*this);
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
  pressio_data compress_typed (pressio_data&& input_data) {
    auto dtype = input_data.dtype();
#if not PRESSIO_MGARD_VERSION_GREATEREQ(0,0,0,3)
    auto itype = dtype_to_itype(input_data.dtype());
#endif
    std::vector<int> dims(3);
    for (int i = 0; i < 3; ++i) {
      dims.at(i) = input_data.get_dimension(i);
      if(dims.at(i) == 0) {
        dims[i] = 1;
      }
    }

    int outsize = 0;
    mgard_compression_function function = select_compression_function(dtype);
    unsigned char* compressed_bytes;
    switch(function)
    {
      case mgard_compression_function::tol:
        compressed_bytes = mgard_compress(
#if not PRESSIO_MGARD_VERSION_GREATEREQ(0, 0, 0, 3)
            itype,
#endif
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
#if not PRESSIO_MGARD_VERSION_GREATEREQ(0, 0, 0, 3)
            itype,
#endif
            static_cast<InputType*>(input_data.data()),
            outsize /*output parameter*/,
            dims.at(0),
            dims.at(1),
            dims.at(2),
            get_converted<InputType>(tolerance),
            get_converted<InputType>(s)
          );
        break;
#if not PRESSIO_MGARD_VERSION_GREATEREQ(0, 0, 0, 3)
      case mgard_compression_function::tol_qoi_s:
        using qoi_fn =  InputType (*)(int, int, int, InputType*);
        compressed_bytes = mgard_compress(
            itype,
            static_cast<InputType*>(input_data.data()),
            outsize /*output parameter*/,
            dims.at(0),
            dims.at(1),
            dims.at(2),
            get_converted<InputType>(tolerance),
            reinterpret_cast<qoi_fn>(qoi_typed(dtype).get_value<void*>()),
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
#else
        /* in the version of mgard after 0.0.0.3 these functions have a different signature */
      case mgard_compression_function::tol_qoi_s:
        if(recompute_metric || dims != old_dims) {
          pressio_mgard_metrics_data data;
          old_dims = dims;

          auto begin = std::chrono::high_resolution_clock::now();
          if(qoi_typed_v(dtype)) {
          //recompute the norm_of_qoi if needed
            data.metric = qoi_metric_name;
            data.metrics = get_metrics();
            //trick the metric plugin by giving it the decompressed data
            data.metrics->begin_compress(&input_data, nullptr);
            
            using qoi_fn_v =  InputType (*)(int, int, int, InputType*, void*);
            norm_of_qoi = mgard::norm(
                dims.at(0), dims.at(1), dims.at(2),
                reinterpret_cast<qoi_fn_v>(qoi_typed_v(dtype).value()),
                get_converted<InputType>(s), reinterpret_cast<void*>(&data));
          } else if(qoi_typed(dtype)) {
            using qoi_fn =  InputType (*)(int, int, int, InputType*);
            norm_of_qoi = mgard::norm(
                dims.at(0), dims.at(1), dims.at(2),
                mgard::QuantityWithoutData<InputType>(reinterpret_cast<qoi_fn>(qoi_typed(dtype).value())),
                get_converted<InputType>(s), nullptr);

          } else {
            set_error(5, "invalid configuration");
            return pressio_data();
          }
          auto end = std::chrono::high_resolution_clock::now();
          training_time = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
          
        }
        //fallthrough
      case mgard_compression_function::tol_normqoi_s:
        compressed_bytes = mgard_compress(
            static_cast<InputType*>(input_data.data()),
            outsize,
            dims.at(0),
            dims.at(1),
            dims.at(2),
            get_converted<InputType>(tolerance),
            get_converted<InputType>(norm_of_qoi),
            get_converted<InputType>(s)
            );
        break;
#endif
      case mgard_compression_function::invalid:
        compressed_bytes = nullptr;
        outsize = 0;
    }

    return pressio_data::move(
        pressio_byte_dtype,
        compressed_bytes,
        std::vector<size_t>{static_cast<size_t>(outsize)},
        pressio_data_libc_free_fn,
        nullptr
        );
  }
  
  template <class InputType>
  int decompress_typed (pressio_data&& input_data, pressio_data* output_data) const {
    mgard_decompression_function function = select_decompression_function();
#if not PRESSIO_MGARD_VERSION_GREATEREQ(0, 0, 0, 3)
    auto itype = dtype_to_itype(output_data->dtype());
    InputType quantizer = 0; /*unused by the mgard as far as I can tell, but part of the signature*/
#endif
    InputType* output_buffer = nullptr;
    std::vector<int> dims(3);
    for (int i = 0; i < 3; ++i) {
      dims[i] = output_data->get_dimension(i);
      if(dims[i] == 0) {
        dims[i] = 1;
      }
    }

    switch(function)
    {
      case mgard_decompression_function::s:
        output_buffer = mgard_decompress(
#if not PRESSIO_MGARD_VERSION_GREATEREQ(0, 0, 0, 3)
            itype,
            quantizer,
#endif
            static_cast<unsigned char*>(input_data.data()),
            static_cast<int32_t>(input_data.size_in_bytes()),
            dims.at(0),
            dims.at(1),
            dims.at(2),
            get_converted<InputType>(s)
            );
        break;
      case mgard_decompression_function::no_s:
        output_buffer = mgard_decompress<InputType>(
#if not PRESSIO_MGARD_VERSION_GREATEREQ(0, 0, 0, 3)
            itype,
            quantizer,
#endif
            static_cast<unsigned char*>(input_data.data()),
            static_cast<int32_t>(input_data.size_in_bytes()),
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
    if(tolerance && !s && !(qoi_typed_v(type) || qoi_typed(type)) && !norm_of_qoi) return mgard_compression_function::tol;
    else if(tolerance && s && !(qoi_typed_v(type) || qoi_typed(type)) && !norm_of_qoi) return mgard_compression_function::tol_s;
    else if(tolerance && s && (qoi_typed(type) || qoi_typed_v(type))) return mgard_compression_function::tol_qoi_s;
    else if(tolerance && s && norm_of_qoi) return mgard_compression_function::tol_normqoi_s;
    else return mgard_compression_function::invalid;
  }

  /**
   * select a decompression function
   * \returns which mgard function should be used for compression
   */
  mgard_decompression_function select_decompression_function() const {
    if(s) return mgard_decompression_function::s;
    else return mgard_decompression_function::no_s;
  }


#if not PRESSIO_MGARD_VERSION_GREATEREQ(0,0,0,3)
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
#endif

  /**
   * helper function that returns the appropriate qoi function pointer
   *
   * \param[in] type the type to use for compression
   * \returns which qoi_function to use based on type
   */
  compat::optional<void*> const& qoi_typed(pressio_dtype type) const {
    assert(type == pressio_double_dtype || type == pressio_float_dtype);
    if(type == pressio_double_dtype) return qoi_double;
    else return qoi_float;
  }

  /**
   * helper function that returns the appropriate qoi function pointer
   *
   * \param[in] type the type to use for compression
   * \returns which qoi_function to use based on type
   */
  compat::optional<void*> const& qoi_typed_v(pressio_dtype type) const {
    assert(type == pressio_double_dtype || type == pressio_float_dtype);
    if(type == pressio_double_dtype) return qoi_double_v;
    else return qoi_float_v;
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
    if(tolerance)
      ss << "tolerance, ";
    if(s)
      ss << "s, ";
    if(qoi_float)
      ss << "qoi_float, ";
    if(qoi_double)
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

#if PRESSIO_MGARD_VERSION_GREATEREQ(0,0,0,3)
  int qoi_use_metric = 0;
  bool recompute_metric = true;
  std::vector<int> old_dims;
  std::string qoi_metric_name;
  compat::optional<unsigned int> training_time;
#endif
  compat::optional<double> tolerance;
  compat::optional<double> s;
  compat::optional<void*> qoi_double;
  compat::optional<void*> qoi_float;
  compat::optional<void*> qoi_double_v;
  compat::optional<void*> qoi_float_v;
  compat::optional<double> norm_of_qoi;
};
static pressio_register compressor_mgard_plugin(compressor_plugins(), "mgard", [](){ return compat::make_unique<mgard_plugin>(); });
