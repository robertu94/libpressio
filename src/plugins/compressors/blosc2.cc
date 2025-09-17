#include <vector>
#include <memory>
#include <sstream>
#include <blosc2.h>
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "pressio_options.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "std_compat/memory.h"
#include "cleanup.h"

namespace libpressio { namespace compressors { namespace blosc2_ns {

class blosc2_plugin: public libpressio_compressor_plugin {
  public:
    struct pressio_options get_options_impl() const override {
      struct pressio_options options;
      set(options, "pressio:lossless", clevel);
      set(options, "blosc2:clevel", clevel);
      set(options, "pressio:nthreads", numinternalthreads);
      set(options, "blosc2:numinternalthreads", static_cast<int32_t>(numinternalthreads));
      set(options, "blosc2:blocksize", blocksize);
      set(options, "blosc2:compressor", compressor);
      return options;
    }

    struct pressio_options get_documentation_impl() const override {
      struct pressio_options options;
      set(options, "pressio:description", R"(blosc2 is a collection of lossless compressors optimized to transfer
        data more quickly than a direct memory fetch can preform. More information on blosc2 can be found on its
        [project homepage](https://blosc2.org/pages/blosc2-in-depth/))");
      set(options, "blosc2:clevel", "compression level");
      set(options, "blosc2:clevel:min", "min compression level");
      set(options, "blosc2:clevel:max", "max compression level");
      set(options, "blosc2:numinternalthreads", "number of threads to use internally");
      set(options, "blosc2:blocksize", "what blocksize should blosc2 use?");
      set(options, "blosc2:compressor", "what lossless compressors should blosc2 use");
      return options;
    }

    struct pressio_options get_configuration_impl() const override {
      struct pressio_options options;
      set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
      set(options, "pressio:stability", "stable");
      std::vector<std::string> compiled_compressors;
      std::string s = blosc2_list_compressors();
      
      size_t begin = 0;
      size_t last_comma = s.find_first_of(',');
      while(last_comma != std::string::npos) {
        compiled_compressors.emplace_back(s.substr(begin, last_comma - begin));
        begin = last_comma + 1;
        last_comma = s.find_first_of(',', begin);
      }
      compiled_compressors.emplace_back(s.substr(begin, last_comma - 1));

      set(options, "blosc2:compressor", compiled_compressors);
      set(options, "pressio:lossless:max", 9);
      set(options, "pressio:lossless:min", 0);
      set(options, "blosc2:clevel:max", 9);
      set(options, "blosc2:clevel:min", 0);

      set(options, "predictors:error_dependent", std::vector<std::string>{});
      set(options, "predictors:error_agnostic", std::vector<std::string>{
"pressio:lossless",
"blosc2:clevel",
"blosc2:blocksize",
"blosc2:compressor",
});


      set(options, "predictors:runtime", std::vector<std::string>{
"pressio:nthreads",
"blosc2:numinternalthreads",
});
      
        set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", {}, std::vector<std::string>{"pressio:lossless", "pressio:nthreads"}));

    return options;
    }

    int set_options_impl(struct pressio_options const& options) override {
      get(options, "pressio:lossless", &clevel);
      get(options, "blosc2:clevel", &clevel);
      uint32_t tmp;
      if(get(options, "pressio:nthreads", &tmp) == pressio_options_key_set) {
        if(tmp > 0) {
          numinternalthreads = tmp;
        } else {
          return set_error(1, "number of threads must be positive");
        }
      }
      int32_t itmp;
      if(get(options, "blosc2:numinternalthreads", &itmp) == pressio_options_key_set) {
        if(itmp > 0) {
          numinternalthreads = itmp;
        } else {
          return set_error(1, "number of threads must be positive");
        }
      }
      get(options, "blosc2:blocksize", &blocksize);
      get(options, "blosc2:compressor", &compressor);

      return 0;
    }

    int compress_impl(const pressio_data *real_input, struct pressio_data* real_output) override {
      pressio_data input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
      int typesize = pressio_dtype_size(input.dtype());
      size_t nbytes = 0, destsize = 0;
      pressio_data output = (real_output->has_data()) ? (domain_manager().make_writeable(domain_plugins().build("malloc"), *real_output)) :
          (pressio_data::owning(pressio_byte_dtype, {nbytes + BLOSC2_MAX_OVERHEAD}));


      blosc2_cparams params = BLOSC2_CPARAMS_DEFAULTS;
      params.clevel = clevel;
      params.typesize = typesize; 
      params.compcode = blosc2_compname_to_compcode(compressor.c_str());
      blosc2_context* ctx = blosc2_create_cctx(params);
      auto cleanup_ctx = make_cleanup([ctx]{blosc2_free_ctx(ctx);});

      auto ret = blosc2_compress_ctx(
              ctx, input.data(), static_cast<int32_t>(nbytes), output.data(), static_cast<int32_t>(destsize)
          );
      //deliberately ignoring warnings from reshape since new size guaranteed to be smaller
      size_t compressed_size = ret;
      if(output.reshape({compressed_size}) > 0) { 
        return reshape_error();
      }

      *real_output = std::move(output);

      if (ret > 0) {
        return 0;
      } else {
        return internal_error(ret);
      }
    }

    int decompress_impl(const pressio_data *real_input, struct pressio_data* real_output) override {
      pressio_data input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
      pressio_data output = (real_output->has_data())?(domain_manager().make_writeable(domain_plugins().build("malloc"), *real_output)):(pressio_data::owning(real_output->dtype(), real_output->dimensions()));

      blosc2_dparams params = BLOSC2_DPARAMS_DEFAULTS;
      params.nthreads = static_cast<int16_t>(this->numinternalthreads);
      blosc2_context* ctx = blosc2_create_dctx(params);
      auto cleanup_ctx = make_cleanup([ctx]{ blosc2_free_ctx(ctx);});

      int ret = blosc2_decompress_ctx(
          ctx,
          input.data(),
          static_cast<int32_t>(input.size_in_bytes()), 
          output.data(),
          static_cast<int32_t>(output.size_in_bytes())
          );

      *real_output = std::move(output);

      if(ret >= 0) {
        return 0;
      } else {
        return internal_error(ret);
      }

    }

    int major_version() const override {
      return BLOSC2_VERSION_MAJOR;
    }
    int minor_version() const override {
      return BLOSC2_VERSION_MINOR;
    }
    int patch_version() const override {
      return BLOSC2_VERSION_RELEASE;
    }

    const char* version() const override {
      return BLOSC2_VERSION_STRING;
    }

    const char* prefix() const override {
      return "blosc2";
    }

    std::shared_ptr<libpressio_compressor_plugin> clone() override {
      return compat::make_unique<blosc2_plugin>(*this);
    }


  private:
    int internal_error(int rc) { std::stringstream ss; ss << "internal error " << rc; return set_error(1, ss.str()); }
    int reshape_error() { return set_error(2, "failed to reshape array after compression"); }

    int clevel = 5;
    uint32_t numinternalthreads = 1;
    unsigned int blocksize = 0;
    std::string compressor = BLOSC_BLOSCLZ_COMPNAME;
    
};

pressio_register registration(compressor_plugins(), "blosc2", [](){ return compat::make_unique<blosc2_plugin>(); });

}}}
