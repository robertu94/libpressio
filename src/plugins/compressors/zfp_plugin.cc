#include <vector>
#include <memory>
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/pressio.h"
#include "pressio_options.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "zfp.h"

class zfp_plugin: public libpressio_compressor_plugin {
  public:
    zfp_plugin() {
      zfp = zfp_stream_open(NULL);
      //note the order here is important. zfp_stream_set_omp_{threads,chunk_size}
      //call zfp_stream_set_execution implicitly.  Since we want the execution policy
      //to be serial by default, we need to set it last.  However, we also want the 
      //threads and chunck sizes to be set rather than undefined in the default case so
      //we must set them too
      zfp_stream_set_omp_threads(zfp, 0);
      zfp_stream_set_omp_chunk_size(zfp, 0);
      zfp_stream_set_execution(zfp,  zfp_exec_serial);
    }
    ~zfp_plugin() {
      zfp_stream_close(zfp);
    }

    struct pressio_options* get_options_impl() const override {
      struct pressio_options* options = pressio_options_new();
      pressio_options_set_uinteger(options, "zfp:minbits", zfp->minbits);
      pressio_options_set_uinteger(options, "zfp:maxbits", zfp->maxbits);
      pressio_options_set_uinteger(options, "zfp:maxprec", zfp->maxprec);
      pressio_options_set_integer(options, "zfp:minexp", zfp->minexp);
      pressio_options_set_integer(options, "zfp:execution", zfp_stream_execution(zfp));
      pressio_options_set_uinteger(options, "zfp:omp_threads", zfp_stream_omp_threads(zfp));
      pressio_options_set_uinteger(options, "zfp:omp_chunk_size", zfp_stream_omp_chunk_size(zfp));
      pressio_options_set_type(options, "zfp:precision", pressio_option_uint32_type);
      pressio_options_set_type(options, "zfp:accuracy", pressio_option_double_type);
      pressio_options_set_type(options, "zfp:rate", pressio_option_double_type);
      pressio_options_set_type(options, "zfp:type", pressio_option_uint32_type);
      pressio_options_set_type(options, "zfp:dims", pressio_option_int32_type);
      pressio_options_set_type(options, "zfp:wra", pressio_option_int32_type);
      pressio_options_set_type(options, "zfp:mode", pressio_option_uint32_type);
      return options;
    }

    struct pressio_options* get_configuration_impl() const override {
      struct pressio_options* options = pressio_options_new();
      pressio_options_set_integer(options, "pressio:thread_safe", pressio_thread_safety_multiple);
      return options;
    }

    int set_options_impl(struct pressio_options const* options) override {
      
      //precision, accuracy, and expert mode settings
      if(unsigned int mode; pressio_options_get_uinteger(options, "zfp:mode", &mode) == pressio_options_key_set) {
        zfp_stream_set_mode(zfp, mode);
      } else if(unsigned int precision; pressio_options_get_uinteger(options, "zfp:precision", &precision) == pressio_options_key_set) {
        zfp_stream_set_precision(zfp, precision);
      } else if (double tolerance; pressio_options_get_double(options, "zfp:accuracy", &tolerance) == pressio_options_key_set) {
        zfp_stream_set_accuracy(zfp, tolerance);
      } else if (double rate; pressio_options_get_double(options, "zfp:rate", &rate) == pressio_options_key_set) {
        unsigned int type, dims;
        int wra;
        if(
            pressio_options_get_uinteger(options, "zfp:type", &type) == pressio_options_key_set &&
            pressio_options_get_uinteger(options, "zfp:dims", &dims) == pressio_options_key_set &&
            pressio_options_get_integer(options, "zfp:wra", &wra) == pressio_options_key_set) {
          zfp_stream_set_rate(zfp, rate, (zfp_type)type, dims, wra);
        } else {
          return invalid_rate();
        }
      } else {
        pressio_options_get_uinteger(options, "zfp:minbits", &zfp->minbits);
        pressio_options_get_uinteger(options, "zfp:maxbits", &zfp->maxbits);
        pressio_options_get_uinteger(options, "zfp:maxprec", &zfp->maxprec);
        pressio_options_get_integer(options, "zfp:minexp", &zfp->minexp);
      }

      if(unsigned int execution;pressio_options_get_uinteger(options, "zfp:execution", &execution) == pressio_options_key_set) { 
        zfp_stream_set_execution(zfp, (zfp_exec_policy)execution);
      }
      if(zfp_stream_execution(zfp) == zfp_exec_omp) {
        if(unsigned int threads; pressio_options_get_uinteger(options, "zfp:omp_threads", &threads) == pressio_options_key_set) {
          zfp_stream_set_omp_threads(zfp, threads);
        }
        if(unsigned int chunk_size; pressio_options_get_uinteger(options, "zfp:omp_chunk_size", &chunk_size) == pressio_options_key_set) {
          zfp_stream_set_omp_chunk_size(zfp, chunk_size);
        }
      }

      return 0;
    }

    int compress_impl(const pressio_data *input, struct pressio_data* output) override {

      zfp_field* in_field;
      auto input_copy = pressio_data::clone(*input);
      if(int ret = convert_pressio_data_to_field(&input_copy, &in_field)) {
        return ret;
      }

      //create compressed data buffer and stream
      size_t bufsize = zfp_stream_maximum_size(zfp, in_field);
      void* buffer = malloc(bufsize);
      bitstream* stream = stream_open(buffer, bufsize);
      zfp_stream_set_bit_stream(zfp, stream);
      zfp_stream_rewind(zfp);

      size_t outsize = zfp_compress(zfp, in_field);
      if(outsize != 0) {
        *output = pressio_data::move(pressio_byte_dtype, stream_data(stream), 1, &outsize, pressio_data_libc_free_fn, nullptr);
        zfp_field_free(in_field);
        stream_close(stream);
        return 0;
      } else {
        zfp_field_free(in_field);
        stream_close(stream);
        return compression_failed();
      }

    }

    int decompress_impl(const pressio_data *input, struct pressio_data* output) override {
      size_t size;
      void* ptr = pressio_data_ptr(input, &size);
      bitstream* stream = stream_open(ptr, size);
      zfp_stream_set_bit_stream(zfp, stream);
      zfp_stream_rewind(zfp);

      enum pressio_dtype dtype = pressio_data_dtype(output);
      size_t dim = pressio_data_num_dimensions(output);
      size_t dims[] = {
        pressio_data_get_dimension(output, 0),
        pressio_data_get_dimension(output, 1),
        pressio_data_get_dimension(output, 2),
        pressio_data_get_dimension(output, 3),
      };
      *output = pressio_data::owning(dtype, dim, dims);
      zfp_field* out_field;

      if(int ret = convert_pressio_data_to_field(output, &out_field)) {
        return ret;
      }
      int num_bytes = zfp_decompress(zfp, out_field);
      zfp_field_free(out_field);
      stream_close(stream);

      if(num_bytes == 0) {
        return decompression_failed();
      } else {
        return 0;
      }
    }

    int major_version() const override {
      return ZFP_VERSION_MAJOR;
    }
    int minor_version() const override {
      return ZFP_VERSION_MINOR;
    }
    int patch_version() const override {
      return ZFP_VERSION_PATCH;
    }
    unsigned int codec_version() const {
      return ZFP_CODEC;
    }

    const char* version() const override {
      return ZFP_VERSION_STRING;
    }

  private:
    int invalid_type() { return set_error(1, "invalid_type");}
    int invalid_dimensions() { return set_error(2, "invalid_dimensions");}
    int compression_failed() { return set_error(3, "compression failed");}
    int decompression_failed() { return set_error(4, "decompression failed");}
    int invalid_rate() { return set_error(1, "if you set rate, you must set type, dims, and wra for the rate mode"); }

    int libpressio_type(pressio_data* data, zfp_type* type) {
      switch(pressio_data_dtype(data))
      {
        case pressio_int32_dtype:
          *type = zfp_type_int32;
          break;
        case pressio_int64_dtype:
          *type = zfp_type_int64;
          break;
        case pressio_double_dtype:
          *type = zfp_type_double;
          break;
        case pressio_float_dtype:
          *type = zfp_type_float;
          break;
        default:
          *type = zfp_type_none;
          invalid_type();
      }
      return 0;
    }
    int convert_pressio_data_to_field(struct pressio_data* data, zfp_field** field) {
      zfp_type type;
      void* in_data = pressio_data_ptr(data, nullptr);
      unsigned int r0 = pressio_data_get_dimension(data, 0);
      unsigned int r1 = pressio_data_get_dimension(data, 1);
      unsigned int r2 = pressio_data_get_dimension(data, 2);
      unsigned int r3 = pressio_data_get_dimension(data, 3);
      if(libpressio_type(data, &type)) {
        return invalid_type();
      }
      switch(pressio_data_num_dimensions(data))
      {
        case 1:
          *field = zfp_field_1d(in_data, type, r0);
          break;
        case 2:
          *field = zfp_field_2d(in_data, type, r0, r1);
          break;
        case 3:
          *field = zfp_field_3d(in_data, type, r0, r1, r2);
          break;
        case 4:
          *field = zfp_field_4d(in_data, type, r0, r1, r2, r3);
          break;
        default:
          *field = nullptr;
          return invalid_dimensions();
      }
      return 0;
    }
    
    zfp_stream* zfp;
};

static inline pressio_register X(compressor_plugins(), "zfp", [](){ return std::make_unique<zfp_plugin>(); });
