#include <vector>
#include <memory>
#include "liblossy_plugin.h"
#include "lossy_options.h"
#include "lossy_data.h"
#include "zfp.h"

class zfp_plugin: public liblossy_plugin {
  public:
    zfp_plugin() {
      zfp = zfp_stream_open(NULL);
      zfp_stream_set_omp_threads(zfp, 0);
      zfp_stream_set_omp_chunk_size(zfp, 0);
      zfp_stream_set_execution(zfp,  zfp_exec_serial);
    }
    ~zfp_plugin() {
      zfp_stream_close(zfp);
    }

    virtual struct lossy_options* get_options() const override {
      struct lossy_options* options = lossy_options_new();
      lossy_options_set_uinteger(options, "zfp:minbits", zfp->minbits);
      lossy_options_set_uinteger(options, "zfp:maxbits", zfp->maxbits);
      lossy_options_set_uinteger(options, "zfp:maxprec", zfp->maxprec);
      lossy_options_set_integer(options, "zfp:minexp", zfp->minexp);
      lossy_options_set_integer(options, "zfp:execution", zfp_stream_execution(zfp));
      lossy_options_set_uinteger(options, "zfp:omp_threads", zfp_stream_omp_threads(zfp));
      lossy_options_set_uinteger(options, "zfp:omp_chunk_size", zfp_stream_omp_chunk_size(zfp));
      lossy_options_clear(options, "zfp:precision");
      lossy_options_clear(options, "zfp:accuracy");
      lossy_options_clear(options, "zfp:rate");
      lossy_options_clear(options, "zfp:type");
      lossy_options_clear(options, "zfp:dims");
      lossy_options_clear(options, "zfp:wra");
      lossy_options_clear(options, "zfp:mode");
      return options;
    }

    virtual int set_options(struct lossy_options const* options) override {
      
      //precision, accuracy, and expert mode settings
      if(unsigned int mode; lossy_options_get_uinteger(options, "zfp:mode", &mode) == lossy_options_key_set) {
        zfp_stream_set_mode(zfp, mode);
      } else if(unsigned int precision; lossy_options_get_uinteger(options, "zfp:precision", &precision) == lossy_options_key_set) {
        zfp_stream_set_precision(zfp, precision);
      } else if (double tolerance; lossy_options_get_double(options, "zfp:accuracy", &tolerance) == lossy_options_key_set) {
        zfp_stream_set_accuracy(zfp, tolerance);
      } else if (double rate; lossy_options_get_double(options, "zfp:rate", &rate) == lossy_options_key_set) {
        unsigned int type, dims, wra;
        if(
            lossy_options_get_uinteger(options, "zfp:type", &type) == lossy_options_key_set &&
            lossy_options_get_uinteger(options, "zfp:dims", &dims) == lossy_options_key_set &&
            lossy_options_get_uinteger(options, "zfp:wra", &wra) == lossy_options_key_set) {
          zfp_stream_set_rate(zfp, rate, (zfp_type)type, dims, wra);
        } else {
          set_error(1, "if you set rate, you must set type, dims, and wra for the rate mode");
          return 1;
        }
      } else {
        lossy_options_get_uinteger(options, "zfp:minbits", &zfp->minbits);
        lossy_options_get_uinteger(options, "zfp:maxbits", &zfp->maxbits);
        lossy_options_get_uinteger(options, "zfp:maxprec", &zfp->maxprec);
        lossy_options_get_integer(options, "zfp:minexp", &zfp->minexp);
      }

      if(unsigned int threads; lossy_options_get_uinteger(options, "zfp:omp_threads", &threads) == lossy_options_key_set) {
        zfp_stream_set_omp_threads(zfp, threads);
      }
      if(unsigned int chunk_size; lossy_options_get_uinteger(options, "zfp:omp_chunk_size", &chunk_size) == lossy_options_key_set) {
        zfp_stream_set_omp_chunk_size(zfp, chunk_size);
      }
      if(unsigned int execution;lossy_options_get_uinteger(options, "zfp:execution", &execution) == lossy_options_key_set) { 
        zfp_stream_set_execution(zfp, (zfp_exec_policy)execution);
      }

      return 0;
    }

    int compress(struct lossy_data* input, struct lossy_data** output) override {

      zfp_field* in_field;
      if(int ret = convert_lossy_data_to_field(input, &in_field)) {
        return ret;
      }

      //create compressed data buffer and stream
      size_t bufsize = zfp_stream_maximum_size(zfp, in_field);
      void* buffer = malloc(bufsize);
      bitstream* stream = stream_open(buffer, bufsize);
      zfp_stream_set_bit_stream(zfp, stream);
      zfp_stream_rewind(zfp);

      size_t outsize = zfp_compress(zfp, in_field);
      lossy_data_free(*output);
      *output = lossy_data_new_move(lossy_byte_dtype, stream_data(stream), 1, &outsize, lossy_data_libc_free_fn, nullptr);

      zfp_field_free(in_field);
      stream_close(stream);
      return 0;
    }

    int decompress(struct lossy_data* input, struct lossy_data** output) override {
      size_t size;
      void* ptr = lossy_data_ptr(input, &size);
      bitstream* stream = stream_open(ptr, size);
      zfp_stream_set_bit_stream(zfp, stream);
      zfp_stream_rewind(zfp);

      enum lossy_dtype dtype = lossy_data_dtype(*output);
      size_t dim = lossy_data_num_dimentions(*output);
      size_t dims[] = {
        lossy_data_get_dimention(*output, 0),
        lossy_data_get_dimention(*output, 1),
        lossy_data_get_dimention(*output, 2),
        lossy_data_get_dimention(*output, 3),
      };
      lossy_data_free(*output);
      *output = lossy_data_new_owning(dtype, dim, dims);
      zfp_field* out_field;

      if(int ret = convert_lossy_data_to_field(*output, &out_field)) {
        return ret;
      }
      zfp_decompress(zfp, out_field);

      zfp_field_free(out_field);
      stream_close(stream);
      return 0;
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
    int invalid_dimentions() { return set_error(2, "invalid_dimentions");}

    int liblossy_type(lossy_data* data, zfp_type* type) {
      switch(lossy_data_dtype(data))
      {
        case lossy_int32_dtype:
          *type = zfp_type_int32;
          break;
        case lossy_int64_dtype:
          *type = zfp_type_int64;
          break;
        case lossy_double_dtype:
          *type = zfp_type_double;
          break;
        case lossy_float_dtype:
          *type = zfp_type_double;
          break;
        default:
          invalid_type();
      }
      return 0;
    }
    int convert_lossy_data_to_field(struct lossy_data* data, zfp_field** field) {
      zfp_type type;
      void* in_data = lossy_data_ptr(data, nullptr);
      unsigned int r0 = lossy_data_get_dimention(data, 0);
      unsigned int r1 = lossy_data_get_dimention(data, 1);
      unsigned int r2 = lossy_data_get_dimention(data, 2);
      unsigned int r3 = lossy_data_get_dimention(data, 3);
      if(liblossy_type(data, &type)) {
        return invalid_type();
      }
      switch(lossy_data_num_dimentions(data))
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
          return invalid_dimentions();
      }
      return 0;
    }
    
    zfp_stream* zfp;
};

std::unique_ptr<liblossy_plugin> make_zfp() {
  return std::make_unique<zfp_plugin>();
}
