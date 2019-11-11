#include <vector>
#include <memory>
#include <sstream>
#include <blosc.h>
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/pressio.h"
#include "pressio_options.h"
#include "pressio_data.h"
#include "pressio_compressor.h"


class blosc_plugin: public libpressio_compressor_plugin {
  public:
    struct pressio_options* get_options_impl() const override {
      struct pressio_options* options = pressio_options_new();
      pressio_options_set_integer(options, "blosc:clevel", clevel);
      pressio_options_set_integer(options, "blosc:numinternalthreads", numinternalthreads);
      pressio_options_set_integer(options, "blosc:doshuffle", doshuffle);
      pressio_options_set_uinteger(options, "blosc:blocksize", blocksize);
      pressio_options_set_string(options, "blosc:compressor", compressor.c_str());
      return options;
    }

    struct pressio_options* get_configuration_impl() const override {
      struct pressio_options* options = pressio_options_new();
      pressio_options_set_integer(options, "pressio:thread_safe", pressio_thread_safety_multiple);
      return options;
    }

    int set_options_impl(struct pressio_options const* options) override {
      pressio_options_get_integer(options, "blosc:clevel", &clevel);
      pressio_options_get_integer(options, "blosc:numinternalthreads", &numinternalthreads);
      pressio_options_get_integer(options, "blosc:doshuffle", &doshuffle);

      unsigned int tmp_blocksize;
      pressio_options_get_uinteger(options, "blosc:blocksize", &tmp_blocksize);
      blocksize = tmp_blocksize;

      const char* tmp = nullptr;
      pressio_options_get_string(options, "blosc:compressor", &tmp);
      compressor = tmp;
      free((void*)tmp);

      return 0;
    }

    int compress_impl(const pressio_data *input, struct pressio_data* output) override {
      int typesize = pressio_dtype_size(pressio_data_dtype(input));
      size_t nbytes = 0, destsize = 0;
      const void* src = pressio_data_ptr(input, &nbytes);
      *output = pressio_data::owning(pressio_byte_dtype, {nbytes + BLOSC_MAX_OVERHEAD});
      void* dest = pressio_data_ptr(output, &destsize);

      auto ret = blosc_compress_ctx(
          clevel,
          doshuffle,
          typesize,
          nbytes,
          src,
          dest,
          destsize,
          compressor.c_str(),
          blocksize,
          numinternalthreads
          );
      //deliberately ignoring warnings from reshape since new size guaranteed to be smaller
      size_t compressed_size = ret;
      if(pressio_data_reshape(output, 1, &compressed_size) > 0) { 
        return reshape_error();
      }

      if (ret > 0) {
        return 0;
      } else {
        return internal_error(ret);
      }
    }

    int decompress_impl(const pressio_data *input, struct pressio_data* output) override {
      const void* src = pressio_data_ptr(input, nullptr);
      if(pressio_data_has_data(output)) {
        std::vector<size_t> dims;
        for (size_t i = 0; i < pressio_data_num_dimensions(output); ++i) {
          dims.push_back(pressio_data_get_dimension(output, i));
        }
        *output = pressio_data::owning(
            pressio_data_dtype(output),
            dims
            );
      }

      size_t destsize;
      void* dest = pressio_data_ptr(output, &destsize);

      int ret = blosc_decompress_ctx(
          src,
          dest,
          destsize,
          numinternalthreads
          );

      if(ret >= 0) {
        return 0;
      } else {
        return internal_error(ret);
      }

    }

    int major_version() const override {
      return BLOSC_VERSION_MAJOR;
    }
    int minor_version() const override {
      return BLOSC_VERSION_MINOR;
    }
    int patch_version() const override {
      return BLOSC_VERSION_RELEASE;
    }

    const char* version() const override {
      return BLOSC_VERSION_STRING;
    }

  private:
    int internal_error(int rc) { std::stringstream ss; ss << "interal error " << rc; return set_error(1, ss.str()); }
    int reshape_error() { return set_error(2, "failed to reshape array after compression"); }

    int clevel;
    int numinternalthreads = 1;
    int doshuffle = BLOSC_NOSHUFFLE;
    size_t blocksize = 0;
    std::string compressor{BLOSC_BLOSCLZ_COMPNAME};
    
};

static pressio_register X(compressor_plugins(), "blosc", [](){ return compat::make_unique<blosc_plugin>(); });

