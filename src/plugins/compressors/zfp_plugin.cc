#include <vector>
#include <memory>
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "pressio_options.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "zfp.h"
#include "std_compat/memory.h"
#include "std_compat/utility.h"

namespace libpressio { namespace zfp_plugin {

class zfp_plugin: public libpressio_compressor_plugin {
  public:
    zfp_plugin() {
      zfp = zfp_stream_open(NULL);
      //note the order here is important. zfp_stream_set_omp_{threads,chunk_size}
      //call zfp_stream_set_execution implicitly.  Since we want the execution policy
      //to be serial by default, we need to set it last.  However, we also want the 
      //threads and chunck sizes to be set rather than undefined in the default case so
      //we must set them too
      zfp_stream_set_accuracy(zfp, 1e-3);
      zfp_stream_set_omp_threads(zfp, 0);
      zfp_stream_set_omp_chunk_size(zfp, 0);
      zfp_stream_set_execution(zfp,  zfp_exec_serial);
    }
    ~zfp_plugin() {
      zfp_stream_close(zfp);
    }
    zfp_plugin(zfp_plugin const& rhs): libpressio_compressor_plugin(rhs), zfp(zfp_stream_open(NULL)) {
      zfp_stream_set_params(zfp, rhs.zfp->minbits, rhs.zfp->maxbits, rhs.zfp->maxprec, rhs.zfp->minexp);
      zfp_stream_set_omp_threads(zfp, zfp_stream_omp_threads(rhs.zfp));
      zfp_stream_set_omp_chunk_size(zfp, zfp_stream_omp_chunk_size(rhs.zfp));
      zfp_stream_set_execution(zfp, zfp_stream_execution(rhs.zfp));
    }
    zfp_plugin(zfp_plugin && rhs) noexcept: zfp(compat::exchange(rhs.zfp, zfp_stream_open(NULL))) {}
    zfp_plugin& operator=(zfp_plugin && rhs) noexcept {
      if(this != &rhs) return *this;
      zfp = compat::exchange(rhs.zfp, zfp_stream_open(NULL));
      return *this;
    }

    zfp_plugin& operator=(zfp_plugin const& rhs) {
      if(this == &rhs) return *this;
      zfp_stream_set_params(zfp, rhs.zfp->minbits, rhs.zfp->maxbits, rhs.zfp->maxprec, rhs.zfp->minexp);
      zfp_stream_set_omp_threads(zfp, zfp_stream_omp_threads(rhs.zfp));
      zfp_stream_set_omp_chunk_size(zfp, zfp_stream_omp_chunk_size(rhs.zfp));
      zfp_stream_set_execution(zfp, zfp_stream_execution(rhs.zfp));
      return *this;
    }

    struct pressio_options get_options_impl() const override {
      struct pressio_options options;
      set_type(options, "pressio:abs", pressio_option_double_type);
      set_type(options, "pressio:lossless", pressio_option_int32_type);
      set(options, "zfp:minbits", zfp->minbits);
      set(options, "zfp:maxbits", zfp->maxbits);
      set(options, "zfp:maxprec", zfp->maxprec);
      set(options, "zfp:minexp", zfp->minexp);
      set(options, "zfp:execution", static_cast<int32_t>(zfp_stream_execution(zfp)));
      set_type(options, "zfp:execution_name", pressio_option_charptr_type);
      set(options, "zfp:omp_threads", zfp_stream_omp_threads(zfp));
      set(options, "zfp:omp_chunk_size", zfp_stream_omp_chunk_size(zfp));
      set_type(options, "zfp:precision", pressio_option_uint32_type);
      set_type(options, "zfp:accuracy", pressio_option_double_type);
      set(options, "zfp:wra", dynamic_wra);
      set(options, "zfp:rate", dynamic_rate);
      set_type(options, "zfp:type", pressio_option_uint32_type);
      set_type(options, "zfp:dims", pressio_option_uint32_type);
      set_type(options, "zfp:mode", pressio_option_uint32_type);
      set_type(options, "zfp:reversible", pressio_option_uint32_type);
      return options;
    }

    struct pressio_options get_documentation_impl() const override {
      struct pressio_options options;
      set(options, "pressio:description", R"(ZFP is an error bounded lossy compressor that uses a transform
          which is similar to a discrete cosine transform. More information on ZFP can be found on its 
          [project homepage](https://zfp.readthedocs.io/en/release0.5.5/))");
      set(options, "zfp:accuracy", "absolute error tolerance for fixed-accuracy mode ");
      set(options, "zfp:dims", "the dimensionality of the input data, used in fixed-rate mode");
      set(options, "zfp:execution", "which execution mode to use");
      set(options, "zfp:execution_name", "which execution mode to use as a human readable string");
      set(options, "zfp:maxbits", "maximum number of bits to store per block");
      set(options, "zfp:maxprec", "maximum number of bit planes to store");
      set(options, "zfp:minbits", "minimum number of bits to store per block");
      set(options, "zfp:minexp", "minimum floating point bit plane number to store");
      set(options, "zfp:mode", "a compact encoding of compressor parameters");
      set(options, "zfp:omp_chunk_size", "OpenMP chunk size used in OpenMP mode");
      set(options, "zfp:omp_threads", "number of OpenMP threads to use in OpenMP mode");
      set(options, "zfp:precision", "The precision specifies how many uncompressed bits per value to store, and indirectly governs the relative error");
      set(options, "zfp:rate", "the rate used in fixed rate mode");
      set(options, "zfp:type", "the type used in fixed rate mode");
      set(options, "zfp:wra", "write random access used in fixed rate mode");
      set(options, "zfp:reversible", "use reversible mode");
      return options;
    }

    struct pressio_options get_configuration_impl() const override {
      struct pressio_options options;
      set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
      set(options, "pressio:stability", "stable");
      set(options, "zfp:execution_name", std::vector<std::string>{"omp", "cuda", "serial"});
      return options;
    }

    int set_options_impl(struct pressio_options const& options) override {
      
      //precision, accuracy, and expert mode settings
      int32_t is_lossless;
      uint32_t mode, precision, reversible; 
      double tolerance, rate; 
      if(get(options, "pressio:abs", &tolerance) == pressio_options_key_set) {
        zfp_stream_set_accuracy(zfp, tolerance);
      } else if(get(options, "pressio:lossless", &is_lossless) == pressio_options_key_set) {
        zfp_stream_set_reversible(zfp);
      }

      if(get(options, "zfp:mode", &mode) == pressio_options_key_set) {
        dynamic_rate = compat::nullopt;
        dynamic_wra = compat::nullopt;
        zfp_stream_set_mode(zfp, mode);
      } else if(get(options, "zfp:precision", &precision) == pressio_options_key_set) {
        dynamic_rate = compat::nullopt;
        dynamic_wra = compat::nullopt;
        zfp_stream_set_precision(zfp, precision);
      } else if (get(options, "zfp:accuracy", &tolerance) == pressio_options_key_set) {
        dynamic_rate = compat::nullopt;
        dynamic_wra = compat::nullopt;
        zfp_stream_set_accuracy(zfp, tolerance);
      } else if (get(options, "zfp:rate", &rate) == pressio_options_key_set) {
        unsigned int type, dims;
        int wra;
        if(
            get(options, "zfp:type", &type) == pressio_options_key_set &&
            get(options, "zfp:dims", &dims) == pressio_options_key_set &&
            get(options, "zfp:wra", &wra) == pressio_options_key_set) {
          dynamic_rate = compat::nullopt;
          zfp_stream_set_rate(zfp, rate, (zfp_type)type, dims, wra);
        } else if(
            !(get(options, "zfp:type", &type) == pressio_options_key_set ||
            get(options, "zfp:dims", &dims) == pressio_options_key_set) &&
            get(options, "zfp:wra", &wra) == pressio_options_key_set)
           {
          dynamic_rate = rate;
          dynamic_wra = wra;
        } else if(
            get(options, "zfp:type", &type) == pressio_options_key_set ||
            get(options, "zfp:dims", &dims) == pressio_options_key_set ||
            get(options, "zfp:wra", &wra) == pressio_options_key_set)
           {
          return invalid_rate();
        } else {
          dynamic_rate = rate;
        }
      } else if (get(options, "zfp:reversible", &reversible) == pressio_options_key_set || get(options, "pressio:lossless", &reversible) == pressio_options_key_set) { 
        zfp_stream_set_reversible(zfp);
      } else {
        dynamic_rate = compat::nullopt;
        get(options, "zfp:minbits", &zfp->minbits);
        get(options, "zfp:maxbits", &zfp->maxbits);
        get(options, "zfp:maxprec", &zfp->maxprec);
        get(options, "zfp:minexp", &zfp->minexp);
      }

      int execution;
      std::string tmp_execution_name;
      if(get(options, "zfp:execution_name", &tmp_execution_name) == pressio_options_key_set) {
        if (tmp_execution_name == "serial") {
          execution = zfp_exec_serial;
        } else if (tmp_execution_name == "omp") {
          execution = zfp_exec_omp;
        } else if (tmp_execution_name == "cuda") {
          execution = zfp_exec_cuda;
        } else {
          return set_error(1, "unknown execution policy: " + tmp_execution_name);
        }
        if(set_execution(static_cast<zfp_exec_policy>(execution))) {
          return error_code();
        }
      } else if(get(options, "zfp:execution", &execution) == pressio_options_key_set) { 
        if(set_execution(static_cast<zfp_exec_policy>(execution))) {
          return error_code();
        }
      }
      if(zfp_stream_execution(zfp) == zfp_exec_omp) {
        unsigned int threads;
        if(get(options, "zfp:omp_threads", &threads) == pressio_options_key_set) {
          zfp_stream_set_omp_threads(zfp, threads);
        }
        unsigned int chunk_size; 
        if(get(options, "zfp:omp_chunk_size", &chunk_size) == pressio_options_key_set) {
          zfp_stream_set_omp_chunk_size(zfp, chunk_size);
        }
      }

      return 0;
    }

    int compress_impl(const pressio_data *input, struct pressio_data* output) override {

      zfp_field* in_field;
      if(int ret = convert_pressio_data_to_field(input, &in_field)) {
        return ret;
      }

      if(dynamic_rate) {
        zfp_type type;
        libpressio_type(input, &type);
        zfp_stream_set_rate(zfp, dynamic_rate.value(), type, input->normalized_dims().size(), dynamic_wra.value_or(0));
      }

      //create compressed data buffer and stream
      size_t bufsize = zfp_stream_maximum_size(zfp, in_field);

      void* buffer;
      bool reuse_buffer;
      if(output->has_data() && output->capacity_in_bytes() >= bufsize) {
        buffer = output->data();
        reuse_buffer = true;
      } else {
        buffer = malloc(bufsize);
        reuse_buffer = false;
      }
      bitstream* stream = stream_open(buffer, bufsize);
      zfp_stream_set_bit_stream(zfp, stream);
      zfp_stream_rewind(zfp);

      size_t outsize = zfp_compress(zfp, in_field);
      if(outsize != 0) {
        if(reuse_buffer) {
          output->set_dtype(pressio_byte_dtype);
          output->reshape({outsize});
        } else {
          *output = pressio_data::move(pressio_byte_dtype, stream_data(stream), 1, &outsize, pressio_data_libc_free_fn, nullptr);
        }
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
      //save the exec mode, set it to serial, and reset it at the end of the decompression
      //if parallel decompression is requested and not supported
      //
      //currently only serial supports all modes, and cuda supports fixed rate decompression
      zfp_exec_policy policy = zfp_stream_execution(zfp);
      zfp_mode zfp_zmode = zfp_stream_compression_mode(zfp);
      if(!  (policy == zfp_exec_serial || (zfp_zmode == zfp_mode_fixed_rate && policy == zfp_exec_cuda))) {
        zfp_stream_set_execution(zfp, zfp_exec_serial);
      }


      size_t size;
      void* ptr = pressio_data_ptr(input, &size);
      bitstream* stream = stream_open(ptr, size);
      zfp_stream_set_bit_stream(zfp, stream);
      zfp_stream_rewind(zfp);

      if(!output->has_data()) {
        enum pressio_dtype dtype = pressio_data_dtype(output);
        size_t dim = pressio_data_num_dimensions(output);
        size_t dims[] = {
          pressio_data_get_dimension(output, 0),
          pressio_data_get_dimension(output, 1),
          pressio_data_get_dimension(output, 2),
          pressio_data_get_dimension(output, 3),
        };
        *output = pressio_data::owning(dtype, dim, dims);
      }
      zfp_field* out_field;

      if(int ret = convert_pressio_data_to_field(output, &out_field)) {
        return ret;
      }
      size_t num_bytes = zfp_decompress(zfp, out_field);
      zfp_field_free(out_field);
      stream_close(stream);

      //reset the execution policy
      zfp_stream_set_execution(zfp, policy);

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

    const char* prefix() const override {
      return "zfp";
    }

    std::shared_ptr<libpressio_compressor_plugin> clone() override{
      return compat::make_unique<zfp_plugin>(*this);
    }


  private:
    int invalid_type() { return set_error(1, "invalid_type");}
    int invalid_dimensions() { return set_error(2, "invalid_dimensions");}
    int compression_failed() { return set_error(3, "compression failed");}
    int decompression_failed() { return set_error(4, "decompression failed");}
    int invalid_rate() { return set_error(1, "if you set rate, you must set type, dims, and wra for the rate mode"); }

    int libpressio_type(pressio_data const* data, zfp_type* type) {
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
    int convert_pressio_data_to_field(struct pressio_data const* data, zfp_field** field) {
      zfp_type type;
      void* in_data = pressio_data_ptr(data, nullptr);
      std::vector<size_t> dims = data->dimensions();
      std::vector<size_t> real_dims;
      std::copy_if(dims.begin(), dims.end(), std::back_inserter(real_dims), [](size_t i){return i != 1;});
      if(libpressio_type(data, &type)) {
        return invalid_type();
      }
      switch(real_dims.size())
      {
        case 1:
          *field = zfp_field_1d(in_data, type, real_dims[0]);
          break;
        case 2:
          *field = zfp_field_2d(in_data, type, real_dims[0], real_dims[1]);
          break;
        case 3:
          *field = zfp_field_3d(in_data, type, real_dims[0], real_dims[1], real_dims[2]);
          break;
        case 4:
          *field = zfp_field_4d(in_data, type, real_dims[0], real_dims[1], real_dims[2], real_dims[3]);
          break;
        default:
          *field = nullptr;
          return invalid_dimensions();
      }
      return 0;
    }

    int set_execution(zfp_exec_policy execution) {
        if(!zfp_stream_set_execution(zfp, execution)) {
          switch ((zfp_exec_policy)execution) {
            case zfp_exec_serial:
              return set_error(1, "zfp serial execution is not available");
            case zfp_exec_omp:
              return set_error(1, "zfp openmp execution is not available");
            case zfp_exec_cuda:
              return set_error(1, "zfp cuda execution is not available");
          }
        }
        return 0;
    }
    
    zfp_stream* zfp;
    compat::optional<double> dynamic_rate;
    compat::optional<int> dynamic_wra;
};

static pressio_register compressor_zfp_plugin(compressor_plugins(), "zfp", [](){ return compat::make_unique<zfp_plugin>(); });

} }
