#include <vector>
#include <memory>
#include <sstream>
#include <blosc.h>
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "pressio_options.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "std_compat/memory.h"

namespace libpressio { namespace blosc {

class blosc_plugin: public libpressio_compressor_plugin {
  public:
    struct pressio_options get_options_impl() const override {
      struct pressio_options options;
      set(options, "pressio:lossless", clevel);
      set(options, "blosc:clevel", clevel);
      set(options, "pressio:nthreads", numinternalthreads);
      set(options, "blosc:numinternalthreads", static_cast<int32_t>(numinternalthreads));
      set(options, "blosc:doshuffle", doshuffle);
      set(options, "blosc:blocksize", blocksize);
      set(options, "blosc:compressor", compressor);
      return options;
    }

    struct pressio_options get_documentation_impl() const override {
      struct pressio_options options;
      set(options, "pressio:description", R"(BLOSC is a collection of lossless compressors optimized to transfer
        data more quickly than a direct memory fetch can preform. More information on BLOSC can be found on its
        [project homepage](https://blosc.org/pages/blosc-in-depth/))");
      set(options, "blosc:clevel", "compression level");
      set(options, "blosc:clevel:min", "min compression level");
      set(options, "blosc:clevel:max", "max compression level");
      set(options, "blosc:numinternalthreads", "number of threads to use internally");
      set(options, "blosc:doshuffle", "should blosc shuffle bits to try to improve compressability");
      set(options, "blosc:blocksize", "what blocksize should blosc use?");
      set(options, "blosc:compressor", "what lossless compressors should blosc use");
      return options;
    }

    struct pressio_options get_configuration_impl() const override {
      struct pressio_options options;
      set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
      set(options, "pressio:stability", "stable");
      std::vector<std::string> compiled_compressors;
      std::string s = blosc_list_compressors();
      
      size_t begin = 0;
      size_t last_comma = s.find_first_of(',');
      while(last_comma != std::string::npos) {
        compiled_compressors.emplace_back(s.substr(begin, last_comma - begin));
        begin = last_comma + 1;
        last_comma = s.find_first_of(',', begin);
      }
      compiled_compressors.emplace_back(s.substr(begin, last_comma - 1));

      set(options, "blosc:compressor", compiled_compressors);
      set(options, "pressio:lossless:max", 9);
      set(options, "pressio:lossless:min", 0);
      set(options, "blosc:clevel:max", 9);
      set(options, "blosc:clevel:min", 0);
      
        std::vector<std::string> invalidations {"pressio:lossless", "blosc:clevel", "blosc:doshuffle", "blosc:blocksize", "blosc:compressor", "pressio:nthreads", "blosc:numinternalthreads"}; 
        std::vector<pressio_configurable const*> invalidation_children {}; 
        
        set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, {}));
        set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
        set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));
        set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{"pressio:lossless", "pressio:nthreads"}));

    return options;
    }

    int set_options_impl(struct pressio_options const& options) override {
      get(options, "pressio:lossless", &clevel);
      get(options, "blosc:clevel", &clevel);
      uint32_t tmp;
      if(get(options, "pressio:nthreads", &tmp) == pressio_options_key_set) {
        if(tmp > 0) {
          numinternalthreads = tmp;
        } else {
          return set_error(1, "number of threads must be positive");
        }
      }
      int32_t itmp;
      if(get(options, "blosc:numinternalthreads", &itmp) == pressio_options_key_set) {
        if(itmp > 0) {
          numinternalthreads = itmp;
        } else {
          return set_error(1, "number of threads must be positive");
        }
      }
      get(options, "blosc:doshuffle", &doshuffle);
      get(options, "blosc:blocksize", &blocksize);
      get(options, "blosc:compressor", &compressor);

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
          static_cast<int32_t>(numinternalthreads)
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
          static_cast<int32_t>(numinternalthreads)
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

    const char* prefix() const override {
      return "blosc";
    }

    std::shared_ptr<libpressio_compressor_plugin> clone() override {
      return compat::make_unique<blosc_plugin>(*this);
    }


  private:
    int internal_error(int rc) { std::stringstream ss; ss << "internal error " << rc; return set_error(1, ss.str()); }
    int reshape_error() { return set_error(2, "failed to reshape array after compression"); }

    int clevel;
    uint32_t numinternalthreads = 1;
    int doshuffle = BLOSC_NOSHUFFLE;
    unsigned int blocksize = 0;
    std::string compressor{BLOSC_BLOSCLZ_COMPNAME};
    
};

static pressio_register compressor_blosc_plugin(compressor_plugins(), "blosc", [](){ return compat::make_unique<blosc_plugin>(); });

}}
